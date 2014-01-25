package blasw

import "github.com/gonum/blas"

var impl blas.Float64
var implCmplx blas.Complex128
var order = blas.RowMajor

func Register(i blas.Float64) {
	impl = i
}

func RegisterCmplx(i blas.Complex128) {
	implCmplx = i
}

func SetOrder(o blas.Order) {
	if o != blas.RowMajor && o != blas.ColMajor {
		panic("blas: illegal order")
	}
	order = o
}
