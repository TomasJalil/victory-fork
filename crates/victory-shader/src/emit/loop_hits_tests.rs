//! `VICTORY_LOOP_HITS` ([`super::loop_hits`]): a loop's latch counts the lanes the loop
//! limit stops, through the root UAV [`super::SLOT_LOOP_HITS`]; without the switch the
//! translation is as before.

use super::{SLOT_LOOP_HITS, lower, validator};
use crate::{io, ir, resource, structure};

/// A loop of 32 threads that counts s6 up until it equals s0 (a root constant: with 0 it
/// goes round four billion times), then stores s6.
fn counted_loop() -> crate::native::NativeProgram {
    super::wave64_tests::native_of(
        &[
            0xBE86_0380, // s_mov_b32 s6, 0
            0x8006_8106, // s_add_u32 s6, s6, 1
            0xBF07_0006, // s_cmp_lg_u32 s6, s0
            0xBF85_FFFD, // s_cbranch_scc1 back to the add
            0x7E00_0206, // v_mov_b32 v0, s6
            0x38 << 26 | 0x1C << 18,
            0x80 << 24 | 2 << 16, // buffer_store_dword v0, off, s[8:11], 0
        ],
        false,
        32,
    )
}

fn lowered(on: bool) -> lower::Lowered {
    let native = counted_loop();
    let p = ir::build(&native.code, native.wave64, native.vgprs).expect("ir::build");
    let st = structure::structure(&p).expect("structure");
    let res = resource::track(&p, &native.layout, native.stage, Some(&st)).expect("resource::track");
    let io = io::raise(&p, &native).expect("io::raise");
    let wave = lower::TEST_HOST_WAVE.replace(Some(32));
    let hits = lower::TEST_LOOP_HITS.replace(Some(on));
    let lowered = lower::lower(&lower::Parts {
        native: &native,
        p: &p,
        st: &st,
        io: &io,
        res: &res,
        inputs: Vec::new(),
        outputs: Vec::new(),
        prims: Vec::new(),
        mesh_stride: 0,
    });
    lower::TEST_LOOP_HITS.set(hits);
    lower::TEST_HOST_WAVE.set(wave);
    lowered.expect("lower")
}

#[test]
fn the_loop_limit_counts_its_lanes_only_when_asked_to() {
    let off = lowered(false);
    assert!(!off.root_uavs.contains(&SLOT_LOOP_HITS));
    // (the program has no wave operation of its own: the count's ballot is the only one)
    assert!(!off.wave_ops);
    let on = lowered(true);
    assert!(on.root_uavs.contains(&SLOT_LOOP_HITS));
    assert!(on.wave_ops);
    assert_ne!(on.dxil, off.dxil);
    if let Ok(mut v) = validator::Validator::load(validator::Validator::default_path().as_deref()) {
        for l in [&off, &on] {
            let mut copy = l.dxil.clone();
            v.validate(&mut copy).expect("DXIL validation failed");
        }
    }
}
