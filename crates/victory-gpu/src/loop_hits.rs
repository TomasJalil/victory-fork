//! `VICTORY_LOOP_HITS=1` ([`victory_shader::emit::loop_hits`]): translated
//! loops add the lanes the loop limit stops to a dword of ours, one per
//! shader, bound as the root UAV `SLOT_LOOP_HITS`. The dwords are taken
//! here once a frame and summed by shader; every few seconds the log gets
//! the shaders with the most, per frame. A lane stopped there computed
//! something else than the console does (it left a loop early, or the
//! program as killed), and the iterations up to the limit are GPU time.
//!
//! shortcut: the dwords are taken while lists may still be adding to them;
//! a count added after the swap is seen the next frame, one added between
//! the read and the clear of the swap (no host-coherent atomics across the
//! bus are assumed) is lost. For a diagnostic either is fine.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::{Duration, Instant};

use windows::Win32::Graphics::Direct3D12::ID3D12Resource;

use crate::host::{HostList, host};
use crate::pipeline::Shader;

/// Shaders with a dword of their own; those after share entry 0.
const ENTRIES: u32 = 4096;
/// Bytes per entry: root views of raw buffers begin on 16 bytes here, as
/// the title's are made to (`USER_VIEW_BIAS`); the first dword is the count.
const STRIDE: u64 = 16;
/// Time between two summaries in the log.
const EVERY: Duration = Duration::from_secs(5);
/// Shaders named in a summary.
const SHOWN: usize = 12;

struct Hits {
    /// The buffer, its mapping, its GPU address.
    buffer: (ID3D12Resource, *mut u32, u64),
    /// By shader hash: its entry.
    entries: HashMap<String, u32>,
    /// Per entry: the shader's hash and stage, and the lanes stopped since
    /// the last summary.
    names: Vec<(String, &'static str, u64)>,
    /// The frame the dwords were last taken in.
    frame: u64,
    /// When the last summary was written, and in which frame.
    said: (Instant, u64),
}

// SAFETY: the mapping is of a buffer the struct owns; all access is under
// the mutex (the GPU's apart).
unsafe impl Send for Hits {}

static HITS: Mutex<Option<Hits>> = Mutex::new(None);

impl Hits {
    fn new() -> Option<Hits> {
        let bytes = u64::from(ENTRIES) * STRIDE;
        let (buffer, data) = host()?.shared_buffer(bytes)?;
        crate::host::name(&buffer, || "loop limit hits".to_owned());
        // SAFETY: a fresh mapping of that many bytes.
        unsafe { std::ptr::write_bytes(data, 0, bytes as usize) };
        // SAFETY: a live buffer.
        let gpu = unsafe { buffer.GetGPUVirtualAddress() };
        tracing::info!("VICTORY_LOOP_HITS: counting the lanes translated loops stop at the loop limit");
        Some(Hits {
            buffer: (buffer, data.cast::<u32>(), gpu),
            entries: HashMap::new(),
            names: vec![("(others)".to_owned(), "", 0)],
            frame: u64::MAX,
            said: (Instant::now(), crate::frame::current()),
        })
    }

    /// `shader`'s entry, given one on first use.
    fn entry(&mut self, shader: &Shader) -> u32 {
        if let Some(&n) = self.entries.get(&shader.hash) {
            return n;
        }
        let n = self.names.len() as u32;
        if n >= ENTRIES {
            return 0;
        }
        self.names.push((shader.hash.clone(), shader.stage, 0));
        self.entries.insert(shader.hash.clone(), n);
        n
    }

    /// Adds what the shaders counted since the last frame to the sums.
    fn take(&mut self) {
        for (n, name) in self.names.iter_mut().enumerate() {
            // SAFETY: inside the mapping, on a dword; the GPU may be adding.
            let at = unsafe { AtomicU32::from_ptr(self.buffer.1.add(n * (STRIDE / 4) as usize)) };
            name.2 += u64::from(at.swap(0, Ordering::Relaxed));
        }
    }

    /// The summary line, every [`EVERY`].
    fn say(&mut self, frame: u64) {
        if self.said.0.elapsed() < EVERY {
            return;
        }
        let frames = frame.saturating_sub(self.said.1).max(1);
        self.said = (Instant::now(), frame);
        let mut hit: Vec<_> = self.names.iter().filter(|n| n.2 > 0).collect();
        hit.sort_unstable_by(|a, b| b.2.cmp(&a.2));
        let total: u64 = hit.iter().map(|n| n.2).sum();
        let list = summary(hit.iter().map(|n| (n.0.as_str(), n.1, n.2)), frames);
        tracing::info!(
            frames,
            shaders = hit.len(),
            lanes = total / frames,
            "loop limit hits per frame: {list}"
        );
        for name in &mut self.names {
            name.2 = 0;
        }
    }
}

/// "<hash> <stage> <lanes per frame>, ..." for the first [`SHOWN`] of
/// `hits` (most first), "none" without any.
fn summary<'a>(hits: impl Iterator<Item = (&'a str, &'a str, u64)>, frames: u64) -> String {
    let mut parts: Vec<String> = Vec::new();
    let mut more = 0;
    for (hash, stage, lanes) in hits {
        if parts.len() == SHOWN {
            more += 1;
            continue;
        }
        parts.push(format!("{hash} {stage} {}", lanes.div_ceil(frames)));
    }
    if parts.is_empty() {
        return "none".to_owned();
    }
    if more > 0 {
        parts.push(format!("and {more} more"));
    }
    parts.join(", ")
}

/// Where `shader` counts in `list`: the GPU address to bind at
/// `SLOT_LOOP_HITS`. None if the host cannot hold the buffer.
pub(crate) fn address(list: &HostList, shader: &Shader) -> Option<u64> {
    let mut guard = HITS.lock().unwrap_or_else(|e| e.into_inner());
    if guard.is_none() {
        *guard = Hits::new();
    }
    let h = guard.as_mut()?;
    let frame = crate::frame::current();
    if h.frame != frame {
        h.frame = frame;
        h.take();
        h.say(frame);
    }
    // Once per list (a key of ours among the textures it brought up to date).
    const KEPT: usize = usize::MAX - 0x6f;
    if list.first_use(KEPT) {
        list.keep(&h.buffer.0);
    }
    let n = h.entry(shader);
    Some(h.buffer.2 + u64::from(n) * STRIDE)
}

#[cfg(test)]
mod tests {
    use super::{SHOWN, summary};

    #[test]
    fn the_summary_names_the_most_hit_shaders_per_frame() {
        assert_eq!(summary(std::iter::empty(), 10), "none");
        let hits = [("aa", "pixel", 100), ("bb", "compute", 15)];
        assert_eq!(summary(hits.into_iter(), 10), "aa pixel 10, bb compute 2");
        let many: Vec<_> = (0..SHOWN as u64 + 3).map(|n| ("cc", "vertex", 1000 - n)).collect();
        assert!(summary(many.into_iter(), 1).ends_with("cc vertex 989, and 3 more"));
    }
}
