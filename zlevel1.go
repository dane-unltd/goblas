package blasw

func Zdotu(x, y VectorCmplx) complex128 {
	must(x.Check())
	must(y.Check())
	if x.N != y.N {
		panic("blas: dimension mismatch")
	}
	return implCmplx.Zdotu(x.N, x.Data, x.Inc, y.Data, y.Inc)
}

func Zdotc(x, y VectorCmplx) complex128 {
	must(x.Check())
	must(y.Check())
	if x.N != y.N {
		panic("blas: dimension mismatch")
	}
	return implCmplx.Zdotc(x.N, x.Data, x.Inc, y.Data, y.Inc)
}

func Znrm2(x VectorCmplx) float64 {
	must(x.Check())
	return implCmplx.Dznrm2(x.N, x.Data, x.Inc)
}

func Dzasum(x VectorCmplx) float64 {
	must(x.Check())
	return implCmplx.Dzasum(x.N, x.Data, x.Inc)
}

func Izmax(x VectorCmplx) int {
	must(x.Check())
	return implCmplx.Izamax(x.N, x.Data, x.Inc)
}

func Zswap(x, y VectorCmplx) {
	must(x.Check())
	must(y.Check())
	if x.N != y.N {
		panic("blas: dimension mismatch")
	}
	implCmplx.Zswap(x.N, x.Data, x.Inc, y.Data, y.Inc)
}

func Zcopy(x, y VectorCmplx) {
	must(x.Check())
	must(y.Check())
	if x.N != y.N {
		panic("blas: dimension mismatch")
	}
	implCmplx.Zcopy(x.N, x.Data, x.Inc, y.Data, y.Inc)
}

func Zaxpy(alpha complex128, x, y VectorCmplx) {
	must(x.Check())
	must(y.Check())
	if x.N != y.N {
		panic("blas: dimension mismatch")
	}
	implCmplx.Zaxpy(x.N, alpha, x.Data, x.Inc, y.Data, y.Inc)
}

func Zscal(alpha complex128, x VectorCmplx) {
	must(x.Check())
	implCmplx.Zscal(x.N, alpha, x.Data, x.Inc)
}

func Zdscal(alpha float64, x VectorCmplx) {
	must(x.Check())
	implCmplx.Zdscal(x.N, alpha, x.Data, x.Inc)
}
