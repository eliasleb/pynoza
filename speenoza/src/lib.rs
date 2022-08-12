pub mod helpers;


pub mod solution {
    use crate::helpers::multi_index::*;
    use std::ops::{Index, IndexMut};
    use std::collections::{HashMap, HashSet};
    use std::hash::Hash;
    use std::cmp::{Eq};
    use ndarray::{Array, Array2, ArrayView2, ArrayViewMut2, ArrayView4, Array4, ArrayViewMut4, Zip,
                  Dim, Ix, Array1, ArrayView1, ArrayViewMut1, Array3, s};
    use std::iter::FromIterator;
    use pyo3::prelude::*;


    pub type Real = f64;

    pub struct Solution {
        max_order: i32,
        aux_fun: HashMap<MultiIndex, HashMap<Signature, Real>>,
    }

    impl Solution {
        pub fn new(max_order: i32) -> Solution {
            let mut aux_fun= HashMap::new();
            aux_fun.insert(
                MultiIndex {
                    i: 0,
                    j: 0,
                    k: 0
                },
                HashMap::from([(Signature { x1: 0, x2: 0, x3: 0, h: 0, r: 1, }, 1.)])
            );
            let mut sol = Solution {
                max_order,
                aux_fun,
            };
            sol.recurse();
            sol
        }

        fn recurse(&mut self) {
            for order in 1..=self.max_order {
                for index in MultiIndexRange::new(MULTI_INDEX_ZERO,
                                                  (self.max_order as u32).try_into().unwrap())
                    .into_iter() {
                    if index.order() != order {
                        continue;
                    }
                    let mut known_index = index.clone();
                    known_index[index.first_non_zero_dim().unwrap().try_into().unwrap()] -= 1;
                    self.increase_order(known_index, index);
                }
            }
        }

        fn increase_order(&mut self, known: MultiIndex, new: MultiIndex) {
            let known_dim = (new - known).first_non_zero_dim().unwrap();
            let known_aux_fun = self.aux_fun.clone();
            let known_aux_fun = known_aux_fun.get(&known).unwrap();
            for (signature, coefficient) in known_aux_fun.iter() {

                // We need to apply the recursion formula to this term.

                // This adds three new terms; two for the space-derivative, and one for the
                // time-derivative

                // First term
                let exponent_x_i = signature[known_dim];
                if exponent_x_i > 0 {
                    let mut identity_first_term = signature.clone();
                    identity_first_term[known_dim] -= 1; // differentiation
                    *self.aux_fun.entry(new)
                        .or_insert(HashMap::new())
                        .entry(identity_first_term)
                        .or_insert(0.) += *coefficient * (exponent_x_i as Real);
                }

                // Second term
                let exponent_r = signature.r;
                let mut identity_second_term = signature.clone();
                identity_second_term[known_dim] += 1; // numerator
                identity_second_term.r += 2; // denominator
                *self.aux_fun.entry(new)
                    .or_insert(HashMap::new())
                    .entry(identity_second_term)
                    .or_insert(0.) -= *coefficient * (exponent_r as Real);

                // Third term (time-derivative)
                let mut identity_third_term = signature.clone();
                identity_third_term[known_dim] += 1; // numerator
                identity_third_term.h += 1; // time-derivative
                identity_third_term.r += 1; // denominator
                *self.aux_fun.entry(new)
                    .or_insert(HashMap::new())
                    .entry(identity_third_term)
                    .or_insert(0.) -= *coefficient;

            }
        }

        pub fn get_human_readable_aux_fun(&self, index: &MultiIndex) -> String {
            Self::term_to_string(self.aux_fun.get(index).unwrap())
        }

        fn term_to_string(term: &HashMap<Signature, Real>) -> String {
            let stuff: Vec<String> = term.iter().map(| (signature, coefficient) |
                String::from(format!(
                    "({}) * {}",
                    coefficient,
                    signature.to_string()
                ))
            ).collect();

            stuff.join(" + ")
        }

        // fn evaluate(&self, index: MultiIndex, t: ArrayView1<Real>, hs: HashMap<i32, Array1<Real>>,
        //             x1: ArrayView1<Real>, x2: ArrayView1<Real>, x3: ArrayView1<Real>,
        //             r: ArrayView1<Real>) -> Array2<Real> {
        //     let mut ret: Array2<Real> = Array2::zeros((t.len(), x1.len()));
        //     for (signature, coefficient,) in self.aux_fun.get(&index)
        //         .unwrap().iter() {
        //         for i_t in 0..t.len() {
        //             for i_x in 0..x1.len() {
        //                 ret[[i_t, i_x]] += hs.get(&signature.h).unwrap()[i_t] * coefficient
        //                 * match signature.x1  {
        //                     pow if pow > 0 => x1[i_x].powi(pow),
        //                     _ => 1.
        //                 } * match signature.x2  {
        //                     pow if pow > 0 => x2[i_x].powi(pow),
        //                     _ => 1.
        //                 } * match signature.x3  {
        //                     pow if pow > 0 => x3[i_x].powi(pow),
        //                     _ => 1.
        //                 } * match signature.r  {
        //                     pow if pow > 0 => 1. / r[i_x].powi(pow),
        //                     _ => 1.
        //                 }
        //             }
        //         }
        //     }
        //     ret
        // }

        pub fn compute_e_field(&self,
                               x1: ArrayView1<Real>,
                               x2: ArrayView1<Real>,
                               x3: ArrayView1<Real>,
                               t: ArrayView1<Real>,
                               h: ArrayView1<Real>,
                               current_moment: ArrayView4<Real>) -> Array3<Real> {
            let r = Self::radius(x1, x2, x3);
            let hs = self.handle_h(t, h);
            let (hs_integral, hs_derivative)
                = self.repack_hs(hs);

            let mut e_field = Array3::zeros((3, t.len(), x1.len()));
            for index in MultiIndexRange::new(MULTI_INDEX_ZERO, self.max_order) {
                let cms = current_moment.slice(s![.., index.i, index.j, index.k]).clone();

                e_field = e_field + self.get_single_term_multipole(index,
                                                                   Array::from_iter(cms.iter().cloned()).view(),
                                                                   t,&hs_derivative, x1, x2, x3, r.view());
            }
            e_field
        }

        pub fn handle_h(&self, t: ArrayView1<Real>, h: ArrayView1<Real>) -> HashMap<i32, Array1<Real>> {
            let mut hs: HashMap<i32, Array1<Real>> = HashMap::new();
            hs.insert(0, h.to_owned());
            let dt = t[1] - t[0];
            hs.insert(-1, Self::antiderivative(dt, h.view()));
            let mut derivative = Self::derivative(dt, h.view());
            for order in 1..=self.max_order + 2 {
                hs.insert(order, derivative.clone());
                derivative = Self::derivative(dt, derivative.view());
            }
            hs
        }

        fn repack_hs(&self, hs: HashMap<i32, Array1<Real>>) -> (HashMap<i32, Array1<Real>>,
                                                                HashMap<i32, Array1<Real>>) {
            let mut hs_integral: HashMap<i32, Array1<Real>> = HashMap::new();
            let mut hs_derivative: HashMap<i32, Array1<Real>> = HashMap::new();
            for order in 0..self.max_order + 1 {
                hs_integral.insert(order, hs.get(&(order - 1)).unwrap().clone());
                hs_derivative.insert(order, hs.get(&(order + 1)).unwrap().clone());
            }

            (hs_integral, hs_derivative)
        }

        fn derivative(dt: Real, x: ArrayView1<Real>) -> Array1<Real> {
            let mut y = Array1::zeros(x.len());
            for (index, yi) in y.iter_mut().enumerate() {
                if index > 0 {
                    *yi = (x[index] - x[index - 1]) / dt;
                }
            }
            y
        }

        fn antiderivative(dt: Real, x: ArrayView1<Real>) -> Array1<Real> {
            let mut y = Array1::zeros(x.len());
            let mut sum: Real = 0.;

            for (i, (yi, xi)) in y.iter_mut().zip(x.iter()).enumerate() {
                *yi = sum;
                sum += xi * dt;
            }
            y
        }

        fn radius(x1: ArrayView1<Real>, x2: ArrayView1<Real>, x3: ArrayView1<Real>) -> Array1<Real> {
            let mut r = Array1::zeros(x1.len());
            Zip::from(&mut r).and(&x1).and(&x2).and(&x3)
                .apply(|r, &x1, &x2, &x3 | {
                    *r = f64::sqrt(x1 * x1 + x2 * x2 + x3 * x3)
                });
            r
        }

        pub fn get_charge_moment(&self, current_moment: ArrayView4<Real>) -> Array4<Real> {
            let dim = (self.max_order + 1) as usize;
            let mut charge_moment: Array4<Real> = Array::zeros((3, dim, dim, dim));
            for i in 0..3 {
                for a in MultiIndexRange::new(MULTI_INDEX_ZERO, self.max_order) {
                    let (a1, a2, a3) = (a.i, a.j, a.k);
                    for j in 0..3 {
                        let mut b = a.clone();
                        if i == j {
                            if a[j] >= 2 {
                                b[j] = a[j] - 2;
                            }
                            charge_moment[[i as usize, a1 as usize, a2 as usize, a3 as usize]] +=
                                f64::from(a[j] * (a[j] - 1)) * current_moment[[j as usize,
                                    b[0] as usize, b[1] as usize, b[2] as usize]];
                        }
                        else {
                            b[i] -= 1;
                            b[j] -= 1;
                            if a[j] >= 1 && a[i] >= 1 {
                                charge_moment[[i as usize, a1 as usize, a2 as usize, a3 as usize]]
                                    += f64::from(a[j] * a[i]) * current_moment[[j as usize,
                                    b[0] as usize, b[1] as usize, b[2] as usize]]
                            }
                        }
                    }
                }
            }
            -charge_moment
        }

        fn get_single_term_multipole(&self, index: MultiIndex, moment: ArrayView1<Real>,
                                     t: ArrayView1<Real>, hs: &HashMap<i32, Array1<Real>>,
                                     x1: ArrayView1<Real>, x2: ArrayView1<Real>,
                                     x3: ArrayView1<Real>, r: ArrayView1<Real>) -> Array3<Real> {
            let mut ret = Array3::zeros((3, t.len(), x1.len()));
            for dim in 0..3 {
                for (signature, coefficient,) in self.aux_fun.get(&index)
                    .unwrap().iter() {
                    for i_t in 0..t.len() {
                        for i_x in 0..x1.len() {
                            ret[[dim, i_t, i_x]] += hs.get(&signature.h).unwrap()[i_t] * coefficient
                                * match signature.x1  {
                                pow if pow > 0 => x1[i_x].powi(pow),
                                _ => 1.
                            } * match signature.x2  {
                                pow if pow > 0 => x2[i_x].powi(pow),
                                _ => 1.
                            } * match signature.x3  {
                                pow if pow > 0 => x3[i_x].powi(pow),
                                _ => 1.
                            } * match signature.r  {
                                pow if pow > 0 => 1. / r[i_x].powi(pow),
                                _ => 1.
                            } * f64::powi(-1., index.order()) / (index.factorial() as f64)
                                * moment[dim];
                        }
                    }
                }
            }
            ret
        }
    }


    impl Signature {
        fn to_string(&self) -> String {
            String::from(format!(
                "{}{}{} {} / {}",
                monomial_to_string("x", self.x1),
                monomial_to_string("y", self.x2),
                monomial_to_string("z", self.x3),
                derivative_to_string("h", self.h),
                monomial_to_string("r", self.r)
            ))
        }
    }

    impl Index<i32> for Signature {
        type Output = i32;
        fn index(&self, index: i32) -> &Self::Output {
            match index {
                0 => &self.x1,
                1 => &self.x2,
                2 => &self.x3,
                _ => panic_on_wrong_index(index)
            }
        }
    }

    impl IndexMut<i32> for Signature {
        fn index_mut(&mut self, index: i32) -> &mut Self::Output {
            match index {
                0 => &mut self.x1,
                1 => &mut self.x2,
                2 => &mut self.x3,
                _ => panic_on_wrong_index(index)
            }
        }
    }

    fn monomial_to_string(var_name: &str, power: i32) -> String {
        match power {
            0 => String::from(""),
            1 => String::from(var_name),
            _ => String::from(format!("{var_name}^{power}"))
        }
    }

    fn derivative_to_string(fun_name: &str, order: i32) -> String {
        match order {
            0 => String::from(fun_name),
            1 => String::from(format!("{fun_name}'",)),
            2 => String::from(format!("{fun_name}''",)),
            3 => String::from(format!("{fun_name}'''",)),
            _ => String::from(format!("{fun_name}^({order})"))
        }
    }

    pub fn check_equality(expr1: &str, expr2: &str) -> bool {
        let lhs: HashSet<&str> = expr1.split("+").map(| x | x.trim() ).collect();
        let rhs: HashSet<&str> = expr2.split("+").map(| x | x.trim() ).collect();

        lhs == rhs
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    struct Signature {
        x1: i32,
        x2: i32,
        x3: i32,
        h: i32,
        r: i32
    }

    /// Formats the sum of two numbers as string.
    #[pyfunction]
    fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
        Ok((a + b).to_string())
    }

    //// A Python module implemented in Rust. The name of this function must match
    //// the `lib.name` setting in the `Cargo.toml`, else Python will not be able to
    //// import the module.
    // #[pymodule]
    // fn speenoza(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    //     //m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    //     Ok(())
    // }
}

#[cfg(test)]
mod tests {
    extern crate csv;
    extern crate ndarray_csv;

    use csv::{ReaderBuilder, WriterBuilder};
    use ndarray_csv::{Array2Reader, Array2Writer};
    use std::error::Error;
    use std::fs::File;
    use super::solution::*;
    use super::helpers::multi_index::{MULTI_INDEX_ZERO, MultiIndex};
    use ndarray::{Array2, Array, s, array, Array4, stack, Array1};
    use itertools::Itertools;

    #[test]
    fn test_auxiliary_function() {
        let sol = Solution::new(1);
        assert!(check_equality(
            &sol.get_human_readable_aux_fun(&MULTI_INDEX_ZERO),
            &String::from("(1) *  h / r")));
        assert!(check_equality(
            &sol.get_human_readable_aux_fun(&MultiIndex { i: 1, j: 0, k: 0 }),
            &String::from("(-1) * x h / r^3 + (-1) * x h' / r^2")
        ));
    }

    #[test]
    fn test_derivatives_and_integral() -> Result<(), Box<dyn Error>> {
        // let sol = Solution::new(2);
        // let t: Array1<Real> =  ndarray::Array::linspace(0., 15., 100);
        // let h = t.mapv(f64::sin);
        // let hs = sol.handle_h(t.view(), h.view());
        // let mut hs_vec: Array2<Real> = Array2::zeros((h.len(), hs.len()));
        // for (i, j) in (0..h.len() as i32).cartesian_product((-1)..(hs.len()-1) as i32) {
        //     hs_vec[[i as usize, (j+1) as usize]] = hs.get(&j).unwrap()[i as usize];
        // }
        //
        // // Write the array into the file.
        // {
        //     let file = File::create("data/test_derivatives_and_integral.csv")?;
        //     let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
        //     writer.serialize_array2(&hs_vec)?;
        // }
        Ok(())
    }

    #[test]
    fn test() {
        let sol = Solution::new(2);
        let x1: Array1<Real> = array![1., 1., 0.];
        let x2: Array1<Real> = array![0., 1., 0.];
        let x3: Array1<Real> = array![0., 0., 1.];
        let t: Array1<Real> =  ndarray::Array::linspace(0., 10., 50);
        let h = t.mapv(f64::sin);
        let mut current_moment = Array4::zeros((3, 5, 5, 5));
        current_moment[[2, 0, 0, 0]] = 1.;
        let e_field = sol.compute_e_field(x1.view(), x2.view(), x3.view(),
                                          t.view(), h.view(), current_moment.view());

    }

    #[test]
    fn test_current_moment_conversion() {
        let max_order = 3;
        let sol = Solution::new(max_order);
        let dim = (max_order + 1) as usize;
        let mut current_moment: Array4<Real> = Array::zeros((3, dim, dim, dim));
        current_moment[[2, 0, 0, 0]] = 1.;
        let charge_moment = sol.get_charge_moment(current_moment.view());
        let mut charge_moment_true: Array4<Real> = Array::zeros((3, dim, dim, dim));
        charge_moment_true[[0, 1, 0, 1]] = -1.;
        charge_moment_true[[1, 0, 1, 1]] = -1.;
        charge_moment_true[[2, 0, 0, 2]] = -2.;
        assert_eq!(charge_moment_true, charge_moment);
    }

}
