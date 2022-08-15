pub mod multi_index {
    extern crate factorial;
    extern crate rayon;

    use std::ops::{Add, Index, IndexMut, Neg, Sub};
    use std::hash::Hash;
    use std::cmp::{Eq};
    use std::fmt::{Display, Formatter};
    use factorial::Factorial;

    pub static MULTI_INDEX_ZERO: MultiIndex = MultiIndex {
        i: 0,
        j: 0,
        k: 0,
    };
    pub static MULTI_INDEX_EI: MultiIndex = MultiIndex {
        i: 1,
        j: 0,
        k: 0,
    };
    pub static MULTI_INDEX_EJ: MultiIndex = MultiIndex {
        i: 0,
        j: 1,
        k: 0,
    };
    pub static MULTI_INDEX_EK: MultiIndex = MultiIndex {
        i: 0,
        j: 0,
        k: 1,
    };

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct MultiIndex {
        pub i: i32,
        pub j: i32,
        pub k: i32
    }

    impl Add for MultiIndex {
        type Output = MultiIndex;

        fn add(self, rhs: Self) -> Self::Output {
            MultiIndex {
                i: self.i + rhs.i,
                j: self.j + rhs.j,
                k: self.k + rhs.k
            }
        }
    }

    impl Neg for MultiIndex {
        type Output = MultiIndex;

        fn neg(self) -> Self::Output {
            MultiIndex {
                i: -self.i,
                j: -self.j,
                k: -self.k
            }
        }
    }

    impl Sub for MultiIndex {
        type Output = MultiIndex;
        fn sub(self, rhs: Self) -> Self::Output {
            MultiIndex {
                i: self.i - rhs.i,
                j: self.j - rhs.j,
                k: self.k - rhs.k
            }
        }
    }

    impl Index<i32> for MultiIndex {
        type Output = i32;

        fn index(&self, index: i32) -> &Self::Output {
            match index {
                0 => &self.i,
                1 => &self.j,
                2 => &self.k,
                _ => panic_on_wrong_index(index),
            }
        }
    }

    impl IndexMut<i32> for MultiIndex {
        fn index_mut(&mut self, index: i32) -> &mut Self::Output {
            match index {
                0 => &mut self.i,
                1 => &mut self.j,
                2 => &mut self.k,
                _ => panic_on_wrong_index(index)
            }
        }
    }

    impl MultiIndex {
        pub fn first_non_zero_dim(&self) -> Result<i32, &'static str> {
            for dim in 0..3 {
                if self[dim] > 0 {
                    return Ok(dim)
                }
            }
            Err("Multi-index has no zero dimension")
        }

        pub fn order(&self) -> i32 {
            self.i + self.j + self.k
        }

        pub fn from_vec(v: Vec<i32>) -> MultiIndex {
            assert!(v.len() >= 3);
            MultiIndex {
                i: v[0],
                j: v[1],
                k: v[2]
            }
        }

        pub fn factorial(&self) -> Result<u64, String> {
            match u64::try_from(self.i.abs()).unwrap().checked_factorial() {
                Some(a) => match u64::try_from(self.j).unwrap().checked_factorial() {
                    Some(b) => match u64::try_from(self.k).unwrap().checked_factorial() {
                        Some(c) => Ok(a * b * c),
                        None => Err(String::from("factorial overflows"))
                    },
                    None => Err(String::from("factorial overflows"))
                },
                None => Err(String::from("factorial overflows"))
            }
        }

        pub fn from<'a, I: IntoIterator<Item=i32>>(is: I, js: I, ks: I) -> Vec<Self> {
            is.into_iter().zip(js.into_iter()).zip(ks.into_iter())
                .map( | ((i, j), k) | MultiIndex {
                    i, j, k
                } ).collect()
        }
    }

    impl Display for MultiIndex {
        fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
            f.write_str(&*format!(
                "({}, {}, {})",
                self.i, self.j, self.k
            ))
        }
    }

    pub struct MultiIndexIterator {
        index: MultiIndex,
        stop: MultiIndex,
        reached_stop: bool,
    }

    impl Iterator for MultiIndexIterator {
        type Item = MultiIndex;

        fn next(&mut self) -> Option<Self::Item> {
            let next_index = match self {
                MultiIndexIterator { index: _, stop: _, reached_stop}
                    if *reached_stop => return None,
                MultiIndexIterator { index, stop, ..}
                    if index == stop => {
                    self.reached_stop = true;
                    return Some(MultiIndex {
                        i: index.i, j: index.j, k: index.k
                    })
                },
                MultiIndexIterator { index: MultiIndex { i: 0, j: 0, k }, .. } => Some(MultiIndex {
                    i: *k + 1,
                    j: 0,
                    k: 0
                }),
                MultiIndexIterator { index: MultiIndex { i: 0, j, k }, .. } => Some(MultiIndex {
                    i: *j - 1,
                    j: 0,
                    k: *k + 1
                }),
                MultiIndexIterator { index: MultiIndex { i, j, k }, .. } => Some(MultiIndex {
                    i: *i - 1,
                    j: *j + 1,
                    k: *k
                }),
            };
            match next_index {
                Some(next_index) => {
                    let current = self.index;
                    self.index = next_index;
                    Some(current)
                }
                _ => None,
            }
        }
    }

    #[derive(Copy, Clone)]
    pub struct MultiIndexRange {
        start: MultiIndex,
        stop: MultiIndex,
    }

    impl MultiIndexRange {
        pub fn new(start: MultiIndex, stop: MultiIndex) -> MultiIndexRange {
            MultiIndexRange {
                start,
                stop
            }
        }

        pub fn stop(order: i32) -> MultiIndex {
            MultiIndex {
                i: 0,
                j: 0,
                k: order
            }
        }

        pub fn splitter(range: Self) -> (Self, Option<Self>) {
            if range.start.order() < range.stop.order() {
                (MultiIndexRange {
                    start: range.start,
                    stop: MultiIndex {
                        i: 0,
                        j: 0,
                        k: range.stop.order() - 1
                    }
                },
                Some(MultiIndexRange {
                    start: MultiIndex {
                        i: range.stop.order(), j: 0, k: 0
                    },
                    stop: range.stop}))
            } else {
                (range, None)
            }
        }
    }

    impl IntoIterator for MultiIndexRange {
        type Item = MultiIndex;
        type IntoIter = MultiIndexIterator;

        fn into_iter(self) -> Self::IntoIter {
            MultiIndexIterator {
                index: self.start,
                stop: self.stop,
                reached_stop: false,
            }
        }
    }

    pub fn panic_on_wrong_index(index: i32) -> ! {
        panic!("given index: {} must be 0, 1 or 2", index)
    }

}
#[cfg(test)]
mod tests {
    use super::multi_index::*;
    use std::iter::*;
    use rayon::prelude::*;
    use std::collections::HashSet;

    #[test]
    fn add_multi_indices() {
        let a1 = MultiIndex { i: 0, j: 0, k: 0 };
        let a2 = MultiIndex { i: 1, j: 2, k: 3 };
        assert_eq!(a1 + a2, a2);
    }

    #[test]
    fn test_iteration() {
        let indices: Vec<MultiIndex> = MultiIndexRange::new(MULTI_INDEX_ZERO, MultiIndexRange::stop(1))
            .into_iter().collect();
        let is = vec![0, 1, 0, 0];
        let js = vec![0, 0, 1, 0];
        let ks = vec![0, 0, 0, 1];
        let indices_true: Vec<MultiIndex> = is.iter().zip(js.iter()).zip(ks.iter())
            .map( | ((i, j), k) | MultiIndex {
                i: *i, j: *j, k: *k
            } ).collect();
        assert_eq!(
            indices,
            indices_true
        );
    }

    #[test]
    fn test_iteration_same_order() {
        let indices: Vec<MultiIndex> = MultiIndexRange::new(MultiIndex {
            i: 2, j: 0, k: 0
        }, MultiIndex{
            i: 0, j: 2, k: 0
        }).into_iter().collect();

        let is = vec![2, 1, 0];
        let js = vec![0, 1, 2];
        let ks = vec![0, 0, 0];
        let indices_true = MultiIndex::from(is, js, ks);
        assert_eq!(
            indices,
            indices_true
        );
    }

    #[test]
    fn test_parallel_iteration() {
        let is = vec![0, 1, 0, 0, 2, 1, 0, 1, 0, 0, ];
        let js = vec![0, 0, 1, 0, 0, 1, 2, 0, 1, 0, ];
        let ks = vec![0, 0, 0, 1, 0, 0, 0, 1, 1, 2, ];
        let indices_true: HashSet<MultiIndex> = HashSet::from_iter(MultiIndex::from(is, js, ks).iter().cloned());
        let indices: HashSet<MultiIndex> = MultiIndexRange::new(
            MULTI_INDEX_ZERO,
            MultiIndexRange::stop(2)
        ).into_iter().par_bridge().collect();

        assert_eq!(
            indices_true,
            indices
        )

    }
}
