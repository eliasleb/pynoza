use speenoza::solution::*;

fn main() {

    for index in MultiIndexRange::new(MULTI_INDEX_ZERO, 10).into_iter() {
        println!("{index}");
    }
}