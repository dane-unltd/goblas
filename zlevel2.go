package blasw

import "github.com/gonum/blas"

func Zgemv(tA blas.Transpose, alpha complex128, A GeneralCmplx, x VectorCmplx, beta complex128, y VectorCmplx) {
	must(A.Check())
	must(x.Check())
	if tA == blas.NoTrans {
		if x.N != A.N {
			panic("blas: dimension mismatch")
		}
	} else if tA == blas.Trans {
		if x.N != A.M {
			panic("blas: dimension mismatch")
		}
	} else {
		panic("blas: illegal value for tA")
	}

	implCmplx.Zgemv(order, tA, A.M, A.N, alpha, A.Data, A.Stride, x.Data, x.Inc, beta, y.Data, y.Inc)
}

func Zgbmv(tA blas.Transpose, alpha complex128, A GeneralCmplxBanded, x VectorCmplx, beta complex128, y VectorCmplx) {
	implCmplx.Zgbmv(order, tA, A.M, A.N, A.KL, A.KU, alpha, A.Data,
		A.Stride, x.Data, x.Inc, beta, y.Data, y.Inc)
}

func Ztrmv(tA blas.Transpose, A TriangularCmplx, x VectorCmplx) {
	implCmplx.Ztrmv(order, A.Uplo, tA, A.Diag, A.N, A.Data, A.Stride, x.Data, x.Inc)
}

func Ztbmv(tA blas.Transpose, A TriangularCmplxBanded, x VectorCmplx) {
	implCmplx.Ztbmv(order, A.Uplo, tA, A.Diag, A.N, A.K, A.Data, A.Stride, x.Data, x.Inc)
}

func Ztpmv(tA blas.Transpose, A TriangularCmplxPacked, x VectorCmplx) {
	implCmplx.Ztpmv(order, A.Uplo, tA, A.Diag, A.N, A.Data, x.Data, x.Inc)
}

func Ztrsv(tA blas.Transpose, A TriangularCmplx, x VectorCmplx) {
	implCmplx.Ztrsv(order, A.Uplo, tA, A.Diag, A.N, A.Data, A.Stride, x.Data, x.Inc)
}

func Ztbsv(tA blas.Transpose, A TriangularCmplxBanded, x VectorCmplx) {
	implCmplx.Ztbsv(order, A.Uplo, tA, A.Diag, A.N, A.K, A.Data, A.Stride, x.Data, x.Inc)
}
func Ztpsv(tA blas.Transpose, A TriangularCmplxPacked, x VectorCmplx) {
	implCmplx.Ztpsv(order, A.Uplo, tA, A.Diag, A.N, A.Data, x.Data, x.Inc)
}

func Zhemv(alpha complex128, A Hermitian, x VectorCmplx, beta complex128, y VectorCmplx) {
	implCmplx.Zhemv(order, A.Uplo, A.N, alpha, A.Data, A.Stride, x.Data, x.Inc, beta, y.Data, y.Inc)
}

func Zhbmv(alpha complex128, A HermitianBanded, x VectorCmplx, beta complex128, y VectorCmplx) {
	implCmplx.Zhbmv(order, A.Uplo, A.N, A.K, alpha, A.Data, A.Stride, x.Data,
		x.Inc, beta, y.Data, y.Inc)
}

func Zhpmv(alpha complex128, A HermitianPacked, x VectorCmplx, beta complex128, y VectorCmplx) {
	implCmplx.Zhpmv(order, A.Uplo, A.N, alpha, A.Data, x.Data, x.Inc, beta, y.Data, y.Inc)
}

func Zgerc(alpha complex128, x VectorCmplx, y VectorCmplx, A GeneralCmplx) {
	implCmplx.Zgerc(order, A.M, A.N, alpha, x.Data, x.Inc, y.Data, y.Inc, A.Data, A.Stride)
}

func Zgeru(alpha complex128, x VectorCmplx, y VectorCmplx, A GeneralCmplx) {
	implCmplx.Zgeru(order, A.M, A.N, alpha, x.Data, x.Inc, y.Data, y.Inc, A.Data, A.Stride)
}

func Zher(alpha float64, x VectorCmplx, A Hermitian) {
	implCmplx.Zher(order, A.Uplo, A.N, alpha, x.Data, x.Inc, A.Data, A.Stride)
}

func Zhpr(alpha float64, x VectorCmplx, A HermitianPacked) {
	implCmplx.Zhpr(order, A.Uplo, A.N, alpha, x.Data, x.Inc, A.Data)
}

func Zher2(alpha complex128, x VectorCmplx, y VectorCmplx, A Hermitian) {
	implCmplx.Zher2(order, A.Uplo, A.N, alpha, x.Data, x.Inc, y.Data, y.Inc, A.Data, A.Stride)
}

func Zhpr2(alpha complex128, x VectorCmplx, y VectorCmplx, A HermitianPacked) {
	implCmplx.Zhpr2(order, A.Uplo, A.N, alpha, x.Data, x.Inc, y.Data, y.Inc, A.Data)
}
