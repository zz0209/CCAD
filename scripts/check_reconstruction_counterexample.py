"""Exact piecewise-polynomial witness: reconstruction need not preserve a query."""
from fractions import Fraction as F
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def integrate_product(a,b,lo,hi):
    # Uniform[-1,1], polynomials represented in ascending powers.
    return sum((ca*cb*(hi**(i+j+1)-lo**(i+j+1))/F(2*(i+j+1)) for i,ca in enumerate(a) for j,cb in enumerate(b)),F(0))

def main():
    breaks=[F(-1),F(0),F(1,2),F(1)];means=[F(0)]*3;second=[[F(0) for _ in range(3)] for _ in range(3)]
    for lo,hi in zip(breaks[:-1],breaks[1:]):
        y=[F(0),F(1)] if lo>=0 else [F(0)];a=[F(-1,2),F(1)] if lo>=F(1,2) else [F(0)];b=[F(1,2),F(-1)] if hi<=F(1,2) else [F(0)];polys=[y,a,b]
        for i,p in enumerate(polys):
            means[i]+=integrate_product(p,[F(1)],lo,hi)
            for j,q in enumerate(polys):second[i][j]+=integrate_product(p,q,lo,hi)
    cov=[[second[i][j]-means[i]*means[j] for j in range(3)] for i in range(3)];aa,ab,bb=cov[1][1],cov[1][2],cov[2][2];cy,dy=cov[0][1],cov[0][2];det=aa*bb-ab*ab;beta=[(bb*cy-ab*dy)/det,(aa*dy-ab*cy)/det];residual=cov[0][0]-beta[0]*cy-beta[1]*dy;ratio=residual/cov[0][0]
    assert beta==[F(3,2),F(-17,54)] and residual==F(11,1296) and ratio==F(11,135)
    payload=dict(distribution='X uniform[-1,1]',source='D=(1,-1), b=0, z=(X_+,(-X)_+)',target='D=(1,-1), b=1/2, z=((X-1/2)_+,(1/2-X)_+)',query='Y=X_+, independently population-centered',means=[str(x) for x in means],covariance=[[str(x) for x in row] for row in cov],optimal_affine_target_coefficients=[str(x) for x in beta],minimum_centered_mse=str(residual),source_variance=str(cov[0][0]),relative_residual=str(ratio),relative_residual_decimal=float(ratio),scope='Exact algebraic counterexample, not trained-seed evidence: same two unit decoder columns, at most one nonzero nonnegative code, both target features alive, both reconstruct X exactly; all target affine readouts still fail to exactly recover source query. Decoder-bias change is explicit.')
    out=ROOT/'artifacts/natural_function_curve_20260906';(out/'reconstruction_counterexample.json').write_text(json.dumps(payload,indent=2)+'\n');print(json.dumps(payload,indent=2))
if __name__=='__main__':main()
