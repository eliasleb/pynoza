pub mod helpers;


pub mod solution {
    use crate::helpers::multi_index::*;
    use std::ops::{Index, IndexMut};
    use std::collections::{HashMap, HashSet};
    use std::hash::Hash;
    use std::cmp::{Eq};
    use ndarray::{Array, ArrayView4, Array4, Zip,
                  Array1, ArrayView1, Array3};
    use pyo3::prelude::*;
    use pyo3::exceptions;
    use numpy::{PyArray3, PyArray1, PyArray4};
    use rayon::prelude::*;

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
                let mut temp = HashMap::new();
                for index in MultiIndexRange::new(
                    MultiIndex { i: order, j: 0, k: 0 },
                    MultiIndexRange::stop(order))
                    .into_iter() {
                    let mut known_index = index.clone();
                    known_index[index.first_non_zero_dim().unwrap().try_into().unwrap()] -= 1;
                    self.increase_order(&mut temp, known_index, index);
                }
                self.aux_fun.extend(temp);
            }
        }

        fn increase_order(&self, temp: &mut HashMap<MultiIndex, HashMap<Signature, Real>>,
                          known: MultiIndex, new: MultiIndex) {
            let known_dim = (new - known).first_non_zero_dim().unwrap();
            let known_aux_fun = self.aux_fun.get(&known).unwrap();
            known_aux_fun
                .iter()
                .for_each(|(signature, coefficient)| {
                    // We need to apply the recursion formula to this term.

                    // This adds three new terms; two for the space-derivative, and one for the
                    // time-derivative

                    // First term
                    let exponent_x_i = signature[known_dim];
                    if exponent_x_i > 0 {
                        let mut identity_first_term = signature.clone();
                        identity_first_term[known_dim] -= 1; // differentiation
                        *temp.entry(new)
                            .or_insert(HashMap::new())
                            .entry(identity_first_term)
                            .or_insert(0.) += *coefficient * (exponent_x_i as Real);
                    }

                    // Second term
                    let exponent_r = signature.r;
                    let mut identity_second_term = signature.clone();
                    identity_second_term[known_dim] += 1; // numerator
                    identity_second_term.r += 2; // denominator
                    *temp.entry(new)
                        .or_insert(HashMap::new())
                        .entry(identity_second_term)
                        .or_insert(0.) -= *coefficient * (exponent_r as Real);

                    // Third term (time-derivative)
                    let mut identity_third_term = signature.clone();
                    identity_third_term[known_dim] += 1; // numerator
                    identity_third_term.h += 1; // time-derivative
                    identity_third_term.r += 1; // denominator
                    *temp.entry(new)
                        .or_insert(HashMap::new())
                        .entry(identity_third_term)
                        .or_insert(0.) -= *coefficient;
                }
                );
        }

        pub fn get_human_readable_aux_fun(&self, index: &MultiIndex, fun_name: String) -> String {
            Self::term_to_string(self.aux_fun.get(index).unwrap(), fun_name)
        }

        fn term_to_string(term: &HashMap<Signature, Real>, fun_name: String) -> String {
            let stuff: Vec<String> = term.iter().map(| (signature, coefficient) |
                String::from(format!(
                    "({}) * {}",
                    coefficient,
                    signature.to_string(fun_name.clone())
                ))
            ).collect();

            stuff.join(" + ")
        }

        pub fn par_compute_e_field(&self,
                               x1: ArrayView1<Real>,
                               x2: ArrayView1<Real>,
                               x3: ArrayView1<Real>,
                               t: ArrayView1<Real>,
                               h: ArrayView1<Real>,
                               current_moment: ArrayView4<Real>) -> Array3<Real> {
            let thresh = 1e-14;
            let r = Self::radius(x1, x2, x3);
            let hs = self.handle_h(t, h);
            let (hs_integral, hs_derivative) = self.repack_hs(hs);
            let charge_moment = self.get_charge_moment(current_moment.view());
            let dt = t[1] - t[0];

            let mut e_field = Array3::zeros((3, t.len(), x1.len()));
            e_field
                .indexed_iter_mut()
                .par_bridge()
                .for_each(
                    |((dim, i_t, i_x), value)| *value = {
                        self.aux_fun.iter()
                            .filter(|(index, _)| {
                                current_moment[[dim, index.i as usize, index.j as usize,
                                    index.k as usize]].abs() > thresh
                                ||  charge_moment[[dim, index.i as usize, index.j as usize,
                                    index.k as usize]].abs() > thresh
                            } )
                            .flat_map(
                                |(index, expression)| {
                                    expression.iter()
                                        .map(
                                            |(signature, coefficient)| {
                                                let i_t_delayed = (i_t as i64) -
                                                    ((r[i_x] / dt) as i64);
                                                coefficient
                                                * x1[i_x].powi(signature.x1)
                                                * x2[i_x].powi(signature.x2)
                                                * x3[i_x].powi(signature.x3)
                                                / r[i_x].powi(signature.r)
                                                * if index.order() % 2 == 0 { 1. } else { -1. }
                                                / index.factorial() * -1e-7 * match i_t_delayed {
                                                    i_t if i_t >= 0 =>
                                                        hs_derivative.get(&signature.h)
                                                        .unwrap()[i_t as usize]
                                                        * current_moment[[dim, index.i as usize,
                                                            index.j as usize, index.k as usize]]
                                                        + hs_integral.get(&signature.h)
                                                        .unwrap()[i_t as usize]
                                                        * charge_moment[[dim, index.i as usize,
                                                            index.j as usize, index.k as usize]],
                                                    _ => 0.,
                                            }
                                        }
                                    )
                                }
                            )
                            .sum::<f64>()
                    }
                );
            e_field
        }


        pub fn compute_electric_field_analytical(&self, current_moment: ArrayView4<Real>)
            -> Vec<String> {
            let thresh = 1e-18;

            let charge_moment = self.get_charge_moment(current_moment.view());
            (0..3)
                .into_iter()
                .map(|dim| {
                    MultiIndexRange::new(
                        MULTI_INDEX_ZERO,
                        MultiIndexRange::stop(self.max_order)
                    )
                        .into_iter()
                        .map(|index| {
                            let current = current_moment[[dim, index.i as usize, index.j as usize, index.k as usize]];
                            let charge = charge_moment[[dim, index.i as usize, index.j as usize, index.k as usize]];
                            let mut element = if current.abs() > thresh || charge.abs() > thresh {
                                String::from(format!(
                                    "+ {} / {} * ",
                                    if index.order() % 2 == 0 { 1 } else { -1 },
                                    index.factorial()
                                ))
                            } else {
                                String::new()
                            };

                            if current.abs() > thresh {
                                element.push_str(&format!(
                                    "{} * ({})",
                                    current,
                                    self.get_human_readable_aux_fun(&index, String::from("h''"))
                                )[..]);
                            }
                            if charge.abs() > thresh {
                                element.push_str(&format!(
                                    "{} * ({})",
                                    charge,
                                    self.get_human_readable_aux_fun(&index, String::from("h"))
                                )[..]);
                            }

                            element
                        })
                        .fold(
                            String::new(),
                                |a, b| String::from(format!(
                                    "{}{}", a, b
                                ))
                        )
                })
                .collect::<Vec<String>>()
        }


        pub fn handle_h(&self, t: ArrayView1<Real>, h: ArrayView1<Real>) -> HashMap<i32, Array1<Real>> {
            let mut hs: HashMap<i32, Array1<Real>> = HashMap::new();
            hs.insert(0, h.to_owned());
            let dt = t[1] - t[0];
            hs.insert(-1, Self::antiderivative(dt, h.view()));
            let mut derivative = Self::derivative(dt, h.view());
            for order in 1..=(self.max_order + 2) {
                hs.insert(order, derivative.clone());
                derivative = Self::derivative(dt, derivative.view());
            }
            hs
        }

        fn repack_hs(&self, hs: HashMap<i32, Array1<Real>>) -> (HashMap<i32, Array1<Real>>,
                                                                HashMap<i32, Array1<Real>>) {
            let mut hs_integral: HashMap<i32, Array1<Real>> = HashMap::new();
            let mut hs_derivative: HashMap<i32, Array1<Real>> = HashMap::new();
            for order in 0..=(self.max_order + 1) {
                hs_integral.insert(order, hs.get(&(order - 1)).unwrap().clone());
                hs_derivative.insert(order, hs.get(&(order + 1)).unwrap().clone());
            }

            (hs_integral, hs_derivative)
        }

        pub fn derivative(dt: Real, x: ArrayView1<Real>) -> Array1<Real> {
            let mut y = Array1::zeros(x.len());
            for (index, yi) in y.iter_mut().enumerate() {
                match index {
                    0 => { *yi = (x[2] - x[0]) / 2. / dt },
                    n if n == x.len() - 1 => { *yi = (x[n] - x[n - 2]) / 2. / dt }
                    n => { *yi = (x[n + 1] - x[n - 1]) / 2. / dt }
                }
            }
            y
        }

        fn antiderivative(dt: Real, x: ArrayView1<Real>) -> Array1<Real> {
            let mut y = Array1::zeros(x.len());
            let mut sum: Real = 0.;

            for (yi, xi) in y.iter_mut().zip(x.iter()) {
                *yi = sum;
                sum += xi * dt;
            }
            y
        }

        fn radius(x1: ArrayView1<Real>, x2: ArrayView1<Real>, x3: ArrayView1<Real>) -> Array1<Real> {
            let mut r = Array1::zeros(x1.len());
            Zip::from(&mut r).and(&x1).and(&x2).and(&x3)
                .for_each(|r, &x1, &x2, &x3 | {
                    *r = f64::sqrt(x1 * x1 + x2 * x2 + x3 * x3)
                });
            r
        }

        pub fn get_charge_moment(&self, current_moment: ArrayView4<Real>) -> Array4<Real> {
            let dim = (self.max_order + 1) as usize;
            let mut charge_moment: Array4<Real> = Array::zeros((3, dim, dim, dim));
            for i in 0..3 {
                for a in MultiIndexRange::new(MULTI_INDEX_ZERO,
                                              MultiIndexRange::stop(self.max_order)) {
                    let (a1, a2, a3) = (a.i, a.j, a.k);
                    for j in 0..3 {
                        let mut b = a.clone();
                        if i == j {
                            if a[j] >= 2 {
                                b[j] = a[j] - 2;
                                charge_moment[[i as usize, a1 as usize, a2 as usize, a3 as usize]]
                                    += f64::from(a[j] * (a[j] - 1))
                                    * current_moment[[j as usize, b[0] as usize, b[1] as usize,
                                    b[2] as usize]];
                            }
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
    }

    impl Signature {
        fn to_string(&self, fun_name: String) -> String {
            String::from(format!(
                "{} {} {} {} / {}",
                monomial_to_string("x", self.x1),
                monomial_to_string("y", self.x2),
                monomial_to_string("z", self.x3),
                derivative_to_string(&fun_name, self.h),
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

    fn convert(x: &PyArray1<Real>) -> ArrayView1<Real> {
        unsafe { x.as_array() }
    }

    #[pyclass]
    struct Speenoza {
        solution: Box<Solution>,
    }

    #[pymethods]
    impl Speenoza {
        #[new]
        fn new(max_order: i32) -> Self {
            let solution = Box::new(Solution::new(max_order));
            Speenoza {
                solution
            }
        }

        fn par_compute_e_field<'a>(&self, x1: &'a PyArray1<Real>, x2: &PyArray1<Real>, x3: &PyArray1<Real>,
                                 t: &PyArray1<Real>, h: &PyArray1<Real>, current_moment: &PyArray4<Real>)
                                 -> PyResult<&'a PyArray3<Real>> {
            let py = x1.py();
            if x2.len() != x1.len() || x3.len() != x1.len() || x2.len() != x3.len() {
                let err: PyErr = PyErr::new::<exceptions::PyValueError, _>(
                    String::from("x1, x2 and x3 must have the same length"));
                return Err(err)
            }
            if t.len() != h.len() {
                let err: PyErr = PyErr::new::<exceptions::PyValueError, _>(
                    String::from("h, t must have the same length"));
                return Err(err)
            }

            if current_moment.shape() != [3usize, (self.solution.max_order + 1) as usize,
                (self.solution.max_order + 1) as usize, (self.solution.max_order + 1) as usize] {
                return Err(PyErr::new::<exceptions::PyValueError, _>(
                    String::from(format!("the shape of current_moment must be (3, dim, dim, dim), \
                    where dim = max_order + 1 (got {:?}, dim should be {})",
                    current_moment.shape(),
                    self.solution.max_order + 1))
                ))
            }

            let x1 = convert(x1);
            let x2 = convert(x2);
            let x3 = convert(x3);
            let t = convert(t);
            let h = convert(h);
            let current_moment = unsafe { current_moment.as_array() };

            let e_field = self.solution.par_compute_e_field(x1, x2, x3, t, h, current_moment);
            let e_field = PyArray3::from_array(py, &e_field);

            Ok(e_field)
        }

    }


    /// Computes the electric field.
    #[pyfunction]
    fn multipole_e_field<'a>(x1: &'a PyArray1<Real>, x2: &PyArray1<Real>, x3: &PyArray1<Real>,
                         t: &PyArray1<Real>, h: &PyArray1<Real>, current_moment: &PyArray4<Real>)
        -> PyResult<&'a PyArray3<Real>> {
        let py = x1.py();
        if x2.len() != x1.len() || x3.len() != x1.len() || x2.len() != x3.len() {
            let err: PyErr = PyErr::new::<exceptions::PyValueError, _>(
                String::from("x1, x2 and x3 must have the same length"));
            return Err(err)
        }
        if t.len() != h.len() {
            let err: PyErr = PyErr::new::<exceptions::PyValueError, _>(
                String::from("h, t must have the same length"));
            return Err(err)
        }

        let max_order = (current_moment.shape()[1] - 1) as i32;
        let sol = Solution::new(max_order);
        let x1 = convert(x1);
        let x2 = convert(x2);
        let x3 = convert(x3);
        let t = convert(t);
        let h = convert(h);
        let current_moment = unsafe { current_moment.as_array() };

        let e_field = sol.par_compute_e_field(x1, x2, x3, t, h, current_moment);
        let e_field = PyArray3::from_array(py, &e_field);

        Ok(e_field)
    }


    /// A Python wrapper around the rust implementation.
    #[pymodule]
    fn speenoza(_py: Python, module: &PyModule) -> PyResult<()> {
        module.add_function(wrap_pyfunction!(multipole_e_field, module)?)?;
        module.add_class::<Speenoza>()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::solution::*;
    use super::helpers::multi_index::{MULTI_INDEX_ZERO, MultiIndex};
    use ndarray::{Array, array, Array4, Array1};

    #[test]
    fn test_auxiliary_function() {
        let sol = Solution::new(2);
        assert!(check_equality(
            &sol.get_human_readable_aux_fun(&MULTI_INDEX_ZERO, String::from("h")),
            &String::from("(1) *    h / r")));
        assert!(check_equality(
            &sol.get_human_readable_aux_fun(&MultiIndex { i: 1, j: 0, k: 0 },
            String::from("h")),
            &String::from("(-1) * x   h' / r^2 + (-1) * x   h / r^3")
        ));
        assert!(check_equality(
            &sol.get_human_readable_aux_fun(&MultiIndex { i: 2, j: 0, k: 0 },
                                            String::from("h")),
            &String::from("(3) * x^2   h' / r^4 + (3) * x^2   h / r^5 + (-1) \
            *    h' / r^2 + (1) * x^2   h'' / r^3 + (-1) *    h / r^3")));
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
        let _e_field = sol.par_compute_e_field(x1.view(), x2.view(), x3.view(),
                                               t.view(), h.view(),
                                               current_moment.view());

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

    #[test]
    fn test_dipole_field() {
        let x1: Array1<Real> = Array1::from(vec![1., -1., 1., -1., 1., -1., 1., -1.]);
        let x2: Array1<Real> = Array1::from(vec![1., 1., -1., -1., 1., 1., -1., -1.]);
        let x3: Array1<Real> = Array1::from(vec![1., 1., 1., 1., -1., -1., -1., -1.]);

        let t: Array1<Real> = Array1::linspace(0f64, 10f64, 500);
        let t0 = 3.;
        let gamma = 1.;
        let h = t.mapv(|t|
            f64::exp(- f64::powi((t - t0) / gamma, 2))
            * (4. * f64::powi((t - t0) / gamma, 2) - 2.)
        );
        let mut current_moment: Array4<Real> = Array4::zeros((3, 3, 3, 3));
        current_moment[[2, 0, 0, 0]] = 1.;
        let sol = Solution::new(2);
        let _e_field = sol.par_compute_e_field(
            x1.view(), x2.view(), x3.view(), t.view(), h.view(),
            current_moment.view());

    }

    #[test]
    fn plot_derivatives() {
        use plotters::prelude::*;
        use plotters::data;

        let t = Array1::linspace(0., 6., 500);
        let dt = t[1] - t[0];
        let t0 = 3.;
        let h = t.mapv(|ti|
            f64::exp(-(ti-t0)*(ti-t0)) * (4. * (ti-t0)*(ti-t0) - 2.));
        let mut derivatives: Vec<Array1<f64>> = Vec::new();
        derivatives.push(h.clone());
        let max_order = 2;
        for order in 1..=max_order {
            derivatives.push(Solution::derivative(dt, derivatives.get(order - 1).unwrap().view()));
        }

        let root_area = BitMapBackend::new("data/time_series.png", (600, 400))
            .into_drawing_area();
        root_area.fill(&WHITE).unwrap();

        let mut ctx = ChartBuilder::on(&root_area)
            .set_label_area_size(LabelAreaPosition::Left, 40)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            // .caption("Scatter Demo", ("sans-serif", 40))
            .build_cartesian_2d(data::fitting_range(t.iter()),
                                data::fitting_range(derivatives
                                    .get(max_order)
                                    .unwrap()
                                    .iter()))
            .unwrap();

        ctx.configure_mesh().draw().unwrap();
        derivatives.iter()
            .for_each(|d| {
                ctx.draw_series(
                    LineSeries::new(t.iter().zip(d).map(|(t, d)| (*t, *d) ), &BLACK)
                ).unwrap();
            });
    }

    #[test]
    fn test_human_readable_field() {
        let sol = Solution::new(2);
        let mut current_moment: Array4<Real> = Array4::zeros((3, 3, 3, 3));
        current_moment[[2, 0, 0, 0]] = 1.;
        let _e_field = sol.compute_electric_field_analytical(current_moment.view());
    }
}
