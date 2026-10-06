"""Independent checks of report formulas. Run: python verify_report.py."""
import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.optimize import brentq


def check():
    x, l, a, b, A, B, T = sp.symbols('x l a b A B T', positive=True)
    kap = sp.sqrt(b) / a
    U = T + (A-T)*sp.cosh(kap*(l-x))/sp.cosh(kap*l) - B/kap*sp.sinh(kap*x)/sp.cosh(kap*l)
    residuals = [a*a*sp.diff(U, x, 2)-b*(U-T), U.subs(x, 0)-A, sp.diff(U, x).subs(x, l)+B]
    assert all(sp.simplify(r) == 0 for r in residuals)
    errors = {'rod': [], 'sphere': [], 'plate': []}
    roots = []
    for n in range(12):
        kn = (n+.5)*np.pi
        def un(s):
            return .4 + .8*np.cosh(np.sqrt(.7)*(1-s))/np.cosh(np.sqrt(.7)) - .3/np.sqrt(.7)*np.sinh(np.sqrt(.7)*s)/np.cosh(np.sqrt(.7))
        numeric = 2*quad(lambda s: (s-un(s))*np.sin(kn*s), 0, 1, epsabs=1e-12)[0]
        exact = 2*((-1)**n/kn**2 - (1.2*kn-.3*(-1)**n+.7*.4/kn)/(kn**2+.7))
        errors['rod'].append(abs(numeric-exact))
        z = brentq(lambda z: np.sin(z)-z*np.cos(z), (n+1)*np.pi+1e-8, (n+1.5)*np.pi-1e-8, xtol=1e-13)
        roots.append(z)
        numerator = quad(lambda s: -(s*s/2-.3)*np.sinc(z*s/np.pi)*s*s, 0, 1, epsabs=1e-13)[0]
        denominator = quad(lambda s: np.sinc(z*s/np.pi)**2*s*s, 0, 1, epsabs=1e-13)[0]
        errors['sphere'].append(abs(numerator/denominator+2/(z*z*np.cos(z))))
        numeric = quad(lambda s: (1-s*s)/2*np.cos(kn*s), -1, 1, epsabs=1e-13)[0]
        errors['plate'].append(abs(numeric-2*(-1)**n/kn**3))
    maximum = {name: max(values) for name, values in errors.items()}
    assert all(value < 1e-10 for value in maximum.values()), maximum
    xx = np.linspace(0, 1, 2001)
    kk = (np.arange(100)+.5)*np.pi
    expected = { .01: (2.8068780654, .4, .9903580317), .06: (1.0555178537, .4475, .7295644344) }
    green = []
    integrate = getattr(np, 'trapezoid', None)
    if integrate is None:
        integrate = np.trapz
    for t, reference in expected.items():
        gg = 2*np.sum(np.sin(kk[:, None]*xx)*np.sin(kk[:, None]*.4)*np.exp(-(kk[:, None]**2+.5)*t), axis=0)
        result = (float(gg.max()), float(xx[gg.argmax()]), float(integrate(gg, xx)))
        assert np.allclose(result, reference, rtol=0, atol=1e-9), result
        green.append((t, result))
    return maximum, roots[:3], green


if __name__ == '__main__':
    maximum, roots, green = check()
    print('Maximum absolute coefficient errors:', maximum)
    print('First three nonzero sphere roots:', roots)
    print('Green kernel (time, (maximum, grid argmax, integral)):', green)
    print('All implemented checks passed. TeX compilation is a separate check.')
