use speenoza::solution::*;

fn main() {
    let range = MultiIndexRange {
        start: MULTI_INDEX_ZERO,
        max_order: 2
    };
    for index in range.into_iter() {
        println!("{index:?}");
    }
}