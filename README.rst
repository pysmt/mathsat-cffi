CFFI Interface for MathSAT 5
============================

The file mathsat_cffi.py provides an interface based on CFFI [1] to
access the MathSAT SMT Solver. This allows the use of the same
"binding" across multiple versions of Python, OS's, and (most 
importantly) with PyPy.

**This is an EXPERIMENTAL interface and will change without notice**



Technical Details
-----------------

The wrapper is composed of two parts. The low-level CFFI interface,
and a middle-level MathSAT wrapper. This distinction is less clear in
the original SWIG wrapper, that mixes the two things. For better or
worse, here we are forced to consider them separately.

The method FFI.cdef (in init_lib) defines which C-level information we
want to access through the CFFI interface. To generate this
information, we copy-paste the API of MathSAT (mathsat.h), and remove
comments. Currently, CFFI is not able to deal with non-constant
MACROS. Therefore we remove those, as well as all functions talking
about types that are not defined there (e.g., gmp).

The rest of the code in init_lib simply tries to find the library
(.so) and open it. If this succeeds, we take the resulting FFI library
and wrap it using the middle-level MathSATWrapper.

In many cases, the functions from the underlying FFI object can be
called directly. In other cases, however, we need to deal with the
return type of the function, and perform some simple conversions. The
class MathSATWrapper takes care of these conversions.


Creating libmathsat.so
----------------------

In the MathSAT distribution [2] there is no .so file. To generate it from
libmathsat.a we perform the following steps:

.. code::

  $ mkdir libmathsat
  $ ar -x ../libmathsat.a
  $ gcc -shared *.o -lgmp -lgmpxx -lstdc++ -o ../libmathsat.so


For the dlopen call of CFFI to work, libmathsat.so must be either in
the same directory as mathsat_cffi.py or in the LD_LIBRARY_PATH.


[1] CFFI: https://bitbucket.org/cffi/cffi/overview

[2] MathSAT 5: http://mathsat.fbk.eu/download.html
