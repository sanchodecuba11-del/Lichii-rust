//! The host GPU's video decode queue (`VK_KHR_video_decode_queue`) and what it decodes.
//!
//! Switch 2 movies (`nx2:vdecsvc`) are AV1, VP9 or H.264. A host GPU with a video decode queue
//! decodes them on its fixed-function video engine, so the CPU does no decoding work. This module
//! finds that queue and reads its AV1 and VP9 decode capabilities.

use std::ffi::{CStr, c_void};

use ash::khr;
use ash::vk;
use parking_lot::Mutex;

use crate::vp9_std;

/// Device extensions every decoder needs, enabled together with at least one codec's.
pub(crate) const DECODE_EXTENSIONS: [&CStr; 2] =
    [khr::video_queue::NAME, khr::video_decode_queue::NAME];

/// The video decode queue and what it decodes.
pub struct VideoQueue {
    /// The queue family the video queue belongs to.
    pub family: u32,
    queue: Mutex<vk::Queue>,
    /// What the queue decodes in AV1 Main, 4:2:0, 8 bits.
    pub av1: Option<Av1Decode>,
    /// What the queue decodes in VP9 profile 0 (4:2:0, 8 bits).
    pub vp9: Option<Vp9Decode>,
}

/// AV1 Main 4:2:0 8-bit decoding on the video queue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Av1Decode {
    /// The highest AV1 level, as `StdVideoAV1Level` (0 is level 2.0, each step a minor level).
    pub max_level: u32,
    /// The queue applies AV1 film grain itself.
    pub film_grain: bool,
    pub caps: DecodeCaps,
}

impl Av1Decode {
    /// The highest level as "major.minor".
    pub fn level_name(&self) -> String {
        format!("{}.{}", 2 + self.max_level / 4, self.max_level % 4)
    }
}

/// VP9 profile 0 decoding on the video queue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vp9Decode {
    /// The highest VP9 level, as `StdVideoVP9Level` (0 is level 1.0, 1 is 1.1, 2 is 2.0, ...).
    pub max_level: u32,
    pub caps: DecodeCaps,
}

impl Vp9Decode {
    /// The highest level as "major.minor".
    pub fn level_name(&self) -> String {
        const NAMES: [&str; 14] = [
            "1.0", "1.1", "2.0", "2.1", "3.0", "3.1", "4.0", "4.1", "5.0", "5.1", "5.2", "6.0",
            "6.1", "6.2",
        ];
        NAMES
            .get(self.max_level as usize)
            .map_or_else(|| format!("#{}", self.max_level), |name| (*name).to_owned())
    }
}

/// What `vkGetPhysicalDeviceVideoCapabilitiesKHR` reports for one decode profile.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DecodeCaps {
    pub min_extent: vk::Extent2D,
    pub max_extent: vk::Extent2D,
    /// Granularity of picture sizes and offsets in decode output and DPB images.
    pub picture_granularity: vk::Extent2D,
    pub max_dpb_slots: u32,
    pub max_active_references: u32,
    /// Alignment of a picture's offset and size in the bitstream buffer.
    pub bitstream_offset_alignment: u64,
    pub bitstream_size_alignment: u64,
    /// Decode output and the reference picture are the same image (one image per picture).
    pub dpb_and_output_coincide: bool,
    /// Decode output and the reference picture are separate images.
    pub dpb_and_output_distinct: bool,
    /// Formats decode output images may have, and DPB images.
    pub output_formats: Vec<vk::Format>,
    pub dpb_formats: Vec<vk::Format>,
    /// The codec's `VK_STD_vulkan_video_codec_*_decode` version sessions are created with.
    pub std_header_version: u32,
}

impl DecodeCaps {
    /// How decode output and reference pictures relate: "shared", "apart" or both.
    pub fn dpb_and_output(&self) -> &'static str {
        match (self.dpb_and_output_coincide, self.dpb_and_output_distinct) {
            (true, true) => "shared or apart",
            (true, false) => "shared",
            _ => "apart",
        }
    }
}

/// The video decode queue family that decodes the most of `wanted`'s codec operations (the
/// first of equals), from each family's flags and video codec operations; `None` when none
/// decodes any of them.
pub(crate) fn decode_family(
    families: &[(vk::QueueFlags, vk::VideoCodecOperationFlagsKHR)],
    wanted: vk::VideoCodecOperationFlagsKHR,
) -> Option<u32> {
    let mut best = None;
    let mut best_count = 0;
    for (index, (flags, operations)) in families.iter().enumerate() {
        let count = (*operations & wanted).as_raw().count_ones();
        if flags.contains(vk::QueueFlags::VIDEO_DECODE_KHR) && count > best_count {
            best = Some(index as u32);
            best_count = count;
        }
    }
    best
}

/// Each queue family's flags and video codec operations. Call only when the device reports
/// `VK_KHR_video_queue`.
pub(crate) fn queue_families(
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> Vec<(vk::QueueFlags, vk::VideoCodecOperationFlagsKHR)> {
    // SAFETY: `physical` came from `instance`, which reports the video queue extension for it.
    unsafe {
        let count = instance.get_physical_device_queue_family_properties2_len(physical);
        let mut video = vec![vk::QueueFamilyVideoPropertiesKHR::default(); count];
        let mut families: Vec<vk::QueueFamilyProperties2<'_>> = video
            .iter_mut()
            .map(|v| vk::QueueFamilyProperties2::default().push_next(v))
            .collect();
        instance.get_physical_device_queue_family_properties2(physical, &mut families);
        let flags: Vec<vk::QueueFlags> = families
            .iter()
            .map(|f| f.queue_family_properties.queue_flags)
            .collect();
        drop(families);
        flags
            .into_iter()
            .zip(video.iter().map(|v| v.video_codec_operations))
            .collect()
    }
}

/// A 4:2:0 8-bit decode profile of `operation` with the codec's profile structure chained.
pub(crate) fn profile_420_8bit(
    operation: vk::VideoCodecOperationFlagsKHR,
    codec_profile: *const c_void,
) -> vk::VideoProfileInfoKHR<'static> {
    let mut profile = vk::VideoProfileInfoKHR::default()
        .video_codec_operation(operation)
        .chroma_subsampling(vk::VideoChromaSubsamplingFlagsKHR::TYPE_420)
        .luma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8)
        .chroma_bit_depth(vk::VideoComponentBitDepthFlagsKHR::TYPE_8);
    profile.p_next = codec_profile;
    profile
}

/// The decode capabilities of `profile` (whose chain names the codec), with `codec_caps`, the
/// codec's capabilities structure, chained to the output; `None` when the device does not decode
/// that profile.
///
/// # Safety
///
/// `codec_caps` points at a valid, writable capabilities structure of `profile`'s codec whose
/// `p_next` is null, and `profile`'s chain is valid.
unsafe fn decode_capabilities(
    video: &khr::video_queue::Instance,
    physical: vk::PhysicalDevice,
    profile: &vk::VideoProfileInfoKHR<'_>,
    codec_caps: *mut c_void,
) -> Option<DecodeCaps> {
    let fp = video.fp();
    let mut decode = vk::VideoDecodeCapabilitiesKHR {
        p_next: codec_caps,
        ..Default::default()
    };
    let mut caps = vk::VideoCapabilitiesKHR {
        p_next: (&mut decode as *mut vk::VideoDecodeCapabilitiesKHR).cast(),
        ..Default::default()
    };
    // SAFETY: a valid profile and output chain; the caller vouches for `codec_caps`.
    let result =
        unsafe { (fp.get_physical_device_video_capabilities_khr)(physical, profile, &mut caps) };
    if result != vk::Result::SUCCESS {
        return None;
    }
    let formats = |usage: vk::ImageUsageFlags| {
        let profiles = [*profile];
        let mut list = vk::VideoProfileListInfoKHR::default().profiles(&profiles);
        let info = vk::PhysicalDeviceVideoFormatInfoKHR::default()
            .image_usage(usage)
            .push_next(&mut list);
        let mut count = 0;
        // SAFETY: the two-call idiom with a valid format info; the output holds `count`.
        unsafe {
            let first = (fp.get_physical_device_video_format_properties_khr)(
                physical,
                &info,
                &mut count,
                std::ptr::null_mut(),
            );
            if first != vk::Result::SUCCESS || count == 0 {
                return Vec::new();
            }
            let mut out = vec![vk::VideoFormatPropertiesKHR::default(); count as usize];
            let second = (fp.get_physical_device_video_format_properties_khr)(
                physical,
                &info,
                &mut count,
                out.as_mut_ptr(),
            );
            if second != vk::Result::SUCCESS && second != vk::Result::INCOMPLETE {
                return Vec::new();
            }
            out.truncate(count as usize);
            out.iter().map(|f| f.format).collect()
        }
    };
    let flags = decode.flags;
    Some(DecodeCaps {
        min_extent: caps.min_coded_extent,
        max_extent: caps.max_coded_extent,
        picture_granularity: caps.picture_access_granularity,
        max_dpb_slots: caps.max_dpb_slots,
        max_active_references: caps.max_active_reference_pictures,
        bitstream_offset_alignment: caps.min_bitstream_buffer_offset_alignment,
        bitstream_size_alignment: caps.min_bitstream_buffer_size_alignment,
        dpb_and_output_coincide: flags
            .contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_COINCIDE),
        dpb_and_output_distinct: flags
            .contains(vk::VideoDecodeCapabilityFlagsKHR::DPB_AND_OUTPUT_DISTINCT),
        output_formats: formats(vk::ImageUsageFlags::VIDEO_DECODE_DST_KHR),
        dpb_formats: formats(vk::ImageUsageFlags::VIDEO_DECODE_DPB_KHR),
        std_header_version: caps.std_header_version.spec_version,
    })
}

/// The AV1 decode capabilities of `physical`, or `None` when it cannot decode AV1 Main 4:2:0
/// 8-bit.
pub(crate) fn av1_capabilities(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> Option<Av1Decode> {
    let video = khr::video_queue::Instance::new(entry, instance);
    let query = |film_grain: bool| {
        let av1_profile = vk::VideoDecodeAV1ProfileInfoKHR::default()
            .std_profile(vk::native::StdVideoAV1Profile_STD_VIDEO_AV1_PROFILE_MAIN)
            .film_grain_support(film_grain);
        let profile = profile_420_8bit(
            vk::VideoCodecOperationFlagsKHR::DECODE_AV1,
            (&av1_profile as *const vk::VideoDecodeAV1ProfileInfoKHR).cast(),
        );
        let mut av1 = vk::VideoDecodeAV1CapabilitiesKHR::default();
        // SAFETY: `av1` is the AV1 capabilities structure, with no chain of its own.
        let caps = unsafe {
            decode_capabilities(
                &video,
                physical,
                &profile,
                (&mut av1 as *mut vk::VideoDecodeAV1CapabilitiesKHR).cast(),
            )
        }?;
        Some(Av1Decode {
            max_level: av1.max_level,
            film_grain,
            caps,
        })
    };
    query(true).or_else(|| query(false))
}

/// The VP9 decode capabilities of `physical`, or `None` when it cannot decode VP9 profile 0.
pub(crate) fn vp9_capabilities(
    entry: &ash::Entry,
    instance: &ash::Instance,
    physical: vk::PhysicalDevice,
) -> Option<Vp9Decode> {
    let video = khr::video_queue::Instance::new(entry, instance);
    let vp9_profile = vp9_std::VideoDecodeVp9ProfileInfo::profile_0();
    let profile = profile_420_8bit(
        vp9_std::DECODE_VP9,
        (&vp9_profile as *const vp9_std::VideoDecodeVp9ProfileInfo).cast(),
    );
    let mut vp9 = vp9_std::VideoDecodeVp9Capabilities::default();
    // SAFETY: `vp9` is the VP9 capabilities structure, with no chain of its own.
    let caps = unsafe {
        decode_capabilities(
            &video,
            physical,
            &profile,
            (&mut vp9 as *mut vp9_std::VideoDecodeVp9Capabilities).cast(),
        )
    }?;
    Some(Vp9Decode {
        max_level: vp9.max_level as u32,
        caps,
    })
}

impl VideoQueue {
    pub(crate) fn new(
        family: u32,
        queue: vk::Queue,
        av1: Option<Av1Decode>,
        vp9: Option<Vp9Decode>,
    ) -> Self {
        Self {
            family,
            queue: Mutex::new(queue),
            av1,
            vp9,
        }
    }

    /// The queue, locked for a submission: video queue submissions are externally synchronised.
    pub fn queue(&self) -> parking_lot::MutexGuard<'_, vk::Queue> {
        self.queue.lock()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_decode_family_decodes_the_most_wanted_codecs() {
        let none = vk::VideoCodecOperationFlagsKHR::empty();
        let av1 = vk::VideoCodecOperationFlagsKHR::DECODE_AV1;
        let families = [
            (vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE, none),
            (
                vk::QueueFlags::VIDEO_DECODE_KHR,
                vk::VideoCodecOperationFlagsKHR::DECODE_H264,
            ),
            (vk::QueueFlags::VIDEO_ENCODE_KHR, av1),
            (
                vk::QueueFlags::VIDEO_DECODE_KHR | vk::QueueFlags::TRANSFER,
                vk::VideoCodecOperationFlagsKHR::DECODE_H264 | av1,
            ),
            (vk::QueueFlags::VIDEO_DECODE_KHR, av1 | vp9_std::DECODE_VP9),
        ];
        let both = av1 | vp9_std::DECODE_VP9;
        assert_eq!(decode_family(&families, av1), Some(3));
        assert_eq!(decode_family(&families, both), Some(4));
        assert_eq!(decode_family(&families, vp9_std::DECODE_VP9), Some(4));
        assert_eq!(decode_family(&families[..3], av1), None);
    }

    #[test]
    fn a_device_decode_queue_reports_usable_capabilities() {
        use crate::device::Device;
        use crate::instance::{Instance, SurfaceSupport};
        use lichii_common::settings::GraphicsSettings;
        use std::sync::Arc;
        let instance = match Instance::new(true, SurfaceSupport::None) {
            Ok(instance) => Arc::new(instance),
            Err(error) => {
                eprintln!("skipping: {error}");
                return;
            }
        };
        let device = match Device::new(instance, &GraphicsSettings::default(), None) {
            Ok(device) => device,
            Err(error) => {
                eprintln!("skipping: {error}");
                return;
            }
        };
        let Some(video) = &device.video else {
            eprintln!("skipping: no video decode queue");
            return;
        };
        assert_ne!(video.family, device.queue_family());
        let all = video.av1.iter().map(|av1| &av1.caps);
        for caps in all.chain(video.vp9.iter().map(|vp9| &vp9.caps)) {
            assert!(
                caps.max_extent.width >= caps.min_extent.width.max(1),
                "{caps:?}"
            );
            assert!(
                caps.max_extent.height >= caps.min_extent.height.max(1),
                "{caps:?}"
            );
            assert!(
                caps.max_dpb_slots > 0 && caps.max_active_references > 0,
                "{caps:?}"
            );
            assert!(
                caps.dpb_and_output_coincide || caps.dpb_and_output_distinct,
                "{caps:?}"
            );
            assert!(
                !caps.output_formats.is_empty() && !caps.dpb_formats.is_empty(),
                "{caps:?}"
            );
        }
        drop(video.queue());
    }

    #[test]
    fn levels_read_as_major_dot_minor() {
        let caps = DecodeCaps {
            min_extent: vk::Extent2D::default(),
            max_extent: vk::Extent2D::default(),
            picture_granularity: vk::Extent2D::default(),
            max_dpb_slots: 0,
            max_active_references: 0,
            bitstream_offset_alignment: 1,
            bitstream_size_alignment: 1,
            dpb_and_output_coincide: false,
            dpb_and_output_distinct: false,
            output_formats: Vec::new(),
            dpb_formats: Vec::new(),
            std_header_version: 0,
        };
        let mut av1 = Av1Decode {
            max_level: 0,
            film_grain: false,
            caps: caps.clone(),
        };
        assert_eq!(av1.level_name(), "2.0");
        av1.max_level = 13;
        assert_eq!(av1.level_name(), "5.1");
        let mut vp9 = Vp9Decode { max_level: 0, caps };
        assert_eq!(vp9.level_name(), "1.0");
        vp9.max_level = 9;
        assert_eq!(vp9.level_name(), "5.1");
    }
}
