# Compare different quadrature rules for integration

There are two examples provided for calculating the weights and abscissas for gaussian quadrature rules, try:

```
make
./gqconstants
```

or

```
python gqconstants.py
```

You can also use the C++ example as a guide to build your own executable

There is no need to look at rules >~25 for Gaussian quadrature.  And you can also stop at ~ 1000 divisions for the trapezoidal and Simpson's rules.  If you run much longer you'll see the numerical errors bevome visible for the trapezoidal, but hyou'll need to think about how to code efficiently or the running time may be very long.

## Error of the integral of exp(-t)
In the algorithmic error regime of the plot, the power law dependence of the relative error on N can be found by estimating the slope of the linear fit of log(error) vs log(N). For the Gaussian method, I get a slope of roughly -24 while for trapezoid and simpson's methods the slope is close to -4 and -1.5 respectively. Clearly the Gaussian method rapidly outpaces the other methods in reducing error as N grows large. I was only able to reach the roundoff regime of the Gaussian and Simpson methods. In this regime, the Gaussian error barely increases as N grows, so its power law relationship is likely a power very close to zero (or roughly a constant error). The Simpson error rapidly increases in this regime but tapers off to roughly a constant error. Therefore, I would say the power law in this case is also very close to zero. 


## Bad Error
The function I chose to produce bad performance is a combination of a highly oscillatory cosine term multiplied by x squared. Because it oscillates very quickly, it takes much larger N (or smaller intervals) before Simpson's method and the trapezoid method can reasonably match the behavior of the function. Before then, they over/underestimate the area under the function greatly. As for the Gaussian method, it takes much higher order polynomials before it can also match how quickly the function oscillates. I think the x squared also adds in some weird behavior that is hard to capture for small N. Really the only way to improve the performance is to minimize the intervals of numerical integration for the Simpson and trapezoid methods, or the order of polynomials in the case of the Gaussian Quadrature method.