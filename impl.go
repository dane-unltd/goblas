package goblas

import "github.com/gonum/blas"

var impl blas.Float64
var implCmplx blas.Complex128

func Register(i blas.Float64) {
	impl = i
}

func RegisterCmplx(i blas.Complex128) {
	implCmplx = i
}
