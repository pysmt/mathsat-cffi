import os
from cffi import FFI
from six import PY2

__all__ = ["ffi", "mathsat"]


def init_lib():
    """Create the Wrapper for the library.

    MATHSAT_MINIMAL_H, and MSATEXIST_ELIM_MINIMAL_H are defined
    below.

    """
    ffi_ = FFI()
    ffi_.cdef(MATHSAT_MINIMAL_H)
    ffi_.cdef(MSATEXIST_ELIM_MINIMAL_H)

    _mathsat = None
    for libname in NAMES:
        try:
            _mathsat = ffi_.dlopen(libname)
        except OSError:
            _mathsat = None
        if _mathsat: break

    if not _mathsat:
        raise ImportError("Cannot find MathSAT library.")
    # Wrap the low-level ffi object
    mathsat_ = MathSATWrapper(_mathsat)
    return ffi_, mathsat_


def demo():
    """Runs a simple demo to test the bindings."""
    # Get the Version of Mathsat
    v_cdata = mathsat.msat_get_version()
    # Convert to string and print
    print(ffi.string(v_cdata))
    # Create a Solver instance
    cfg_ = mathsat.msat_create_config()
    env_ = mathsat.msat_create_env(cfg_)
    a = mathsat.msat_make_number(env_, b"5")
    b = mathsat.msat_make_number(env_, b"5")
    # Show details about how objects are wrapped
    print(type(a))
    print(ffi.typeof(a))
    # At the library level, a and b are different objects
    assert a != b
    # But they point to the same MathSAT term
    # (see msat_term struct)
    assert a.repr == b.repr
    # Assert and Solver
    false_ = mathsat.msat_make_false(env_)
    rv = mathsat.msat_assert_formula(env_, false_)
    assert rv == 0
    res = mathsat.msat_solve(env_)
    # msat_result is a proper enum in the library
    if res == mathsat.MSAT_UNSAT:
        print("UNSAT")
    return


class MathSATWrapper(object):
    """Wrapper to the FFI library object.

    This deals with type conversions and simple macros.

    Attributes defined here override the ones of the FFI library (see
    __getattr__)

    """
    def __init__(self, cffi_lib):
        self.lib = cffi_lib

    def __getattr__(self, name):
        try:
            return super(MathSATWrapper).getattr(self, name)
        except AttributeError:
            return getattr(self.lib, name)

    def MSAT_ERROR_DECL(self, decl):
        return decl.repr == None

    def MSAT_ERROR_CONFIG(self, cfg):
        return cfg.repr == None

    def MSAT_ERROR_ENV(self, env):
        return env.repr == None

    def MSAT_ERROR_TERM(self, term):
        return term.repr == None

    def MSAT_ERROR_MODEL(self, model):
        return model.repr == None

    def MSAT_ERROR_TYPE(self, type_):
        return type_.repr == None

    def MSAT_ERROR_MODEL_ITERATOR(self, iter_):
        return iter_.repr == None

    def msat_get_function_type(self, env, param_types, return_type):
        return self.lib.msat_get_function_type(env, param_types,
                                               len(param_types),
                                               return_type)

    def msat_model_iterator_next(self, i):
        "returns a tuple (term, value)"
        t = ffi.new("msat_term *")
        v = ffi.new("msat_term *")
        self.lib.msat_model_iterator_next(i, t, v)
        return (t[0], v[0])

    def msat_term_repr(self, t):
        msat_str = self.lib.msat_term_repr(t)
        tstr = ffi.string(msat_str)
        self.lib.msat_free(msat_str)
        if not PY2:
            tstr = tstr.decode('ascii')
        return tstr

    def msat_is_array_type(self, env, tp):
        index_type = ffi.new("msat_type *")
        elem_type = ffi.new("msat_type *")
        res = self.lib.msat_is_array_type(env, tp, index_type, elem_type)
        return (res, index_type[0], elem_type[0])

    def msat_is_bv_type(self, env, tp):
        size = ffi.new("size_t *")
        res = self.lib.msat_is_bv_type(env, tp, size)
        return (res, size[0])

    def msat_term_is_bv_zext(self, env, term):
        size = ffi.new("size_t *")
        res = self.lib.msat_term_is_bv_zext(env, term, size)
        return (res, size[0])

    def msat_term_is_bv_sext(self, env, term):
        size = ffi.new("size_t *")
        res = self.lib.msat_term_is_bv_sext(env, term, size)
        return (res, size[0])

    def msat_term_is_bv_rol(self, env, term):
        size = ffi.new("size_t *")
        res = self.lib.msat_term_is_bv_rol(env, term, size)
        return (res, size[0])

    def msat_term_is_bv_ror(self, env, term):
        size = ffi.new("size_t *")
        res = self.lib.msat_term_is_bv_ror(env, term, size)
        return (res, size[0])

    def msat_term_is_bv_extract(self, env, term):
        msb = ffi.new("size_t*")
        lsb = ffi.new("size_t*")
        res = self.lib.msat_term_is_bv_extract(env, term, msb, lsb)
        return (res, msb[0], lsb[0])

    def msat_solve_with_assumptions(self, env, assumptions):
        return self.lib.msat_solve_with_assumptions(env, assumptions,
                                                    len(assumptions))

    def msat_create_env(self, conf=None, other=None):
        try:
            if conf is None:
                cfg = self.lib.msat_create_config()
            elif hasattr(conf, 'iteritems'):
                cfg = self.lib.msat_create_config()
                for (k, v) in conf.iteritems():
                    self.lib.msat_set_option(cfg, k, v)
            elif hasattr(conf, 'read'):
                cfg = self.lib.msat_parse_config(conf.read())
            else:
                try:
                    cfg = conf + ""
                except:
                    cfg = conf
                else:
                    if '=' not in cfg:
                        cfg = self.lib.msat_create_default_config(cfg)
                    else:
                        cfg = self.lib.msat_parse_config(cfg)
            if other is not None:
                return self.lib.msat_create_shared_env(cfg, other)
            else:
                return self.lib.msat_create_env(cfg)
        finally:
            if cfg is not conf:
                self.lib.msat_destroy_config(cfg)

    msat_create_shared_env = msat_create_env

    def msat_get_unsat_core(self, env):
        size = ffi.new("size_t*")
        term_list = self.lib.msat_get_unsat_core(env, size)
        res = [] # TODO: What if term_list is Null?
        for i in range(size[0]):
            n = ffi.new("msat_term*")
            n[0] = term_list[i]
            res.append(n[0])
        self.lib.msat_free(term_list)
        return res

    def msat_get_unsat_assumptions(self, env):
        size = ffi.new("size_t*")
        term_list = self.lib.msat_get_unsat_assumptions(env, size)
        res = [] # TODO: What if term_list is Null?
        for i in range(size[0]):
            n = ffi.new("msat_term*")
            n[0] = term_list[i]
            res.append(n[0])
        self.lib.msat_free(term_list)
        return res

    def msat_create_default_config(self, logic):
        if not PY2:
            logic = logic.encode('ascii')
        return self.lib.msat_create_default_config(logic)

    def msat_set_option(self, config, key, value):
        if not PY2:
            key, value = key.encode('ascii'), value.encode('ascii')
        return self.lib.msat_set_option(config, key, value)

    def msat_make_number(self, env, rep):
        if not PY2:
            rep = rep.encode('ascii')
        return self.lib.msat_make_number(env, rep)

    def msat_make_bv_number(self, env, rep, width, base):
        if not PY2:
            rep = rep.encode('ascii')
        return self.lib.msat_make_bv_number(env, rep, width, base)

    def msat_declare_function(self, env, name, type_):
        if not PY2:
            name = name.encode('ascii')
        return self.lib.msat_declare_function(env, name, type_)

# EOC MathSATWrapper

#
# Define Constants
#
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NAMES = ["libmathsat.so",
          os.path.join(BASE_DIR, "libmathsat.so"),
         ]

# These _H were obtained from MathSAT API .h files
MATHSAT_MINIMAL_H = """

typedef struct msat_config { void *repr; } msat_config;
typedef struct msat_env { void *repr; } msat_env;
typedef struct msat_term { void *repr; } msat_term;
typedef struct msat_decl { void *repr; } msat_decl;
typedef struct msat_type { void *repr; } msat_type;

typedef enum {
    MSAT_UNKNOWN = -1, /**< Unknown. */
    MSAT_UNSAT,        /**< Unsatisfiable. */
    MSAT_SAT           /**< Satisfiable. */
} msat_result;

typedef enum {
    MSAT_UNDEF = -1,  /**< Undefined/unknown. */
    MSAT_FALSE,       /**< False. */
    MSAT_TRUE         /**< True. */
} msat_truth_value;


typedef int (*msat_all_sat_model_callback)(msat_term *model, int size,
                                           void *user_data);
typedef struct msat_model { void *repr; } msat_model;
typedef struct msat_model_iterator { void *repr; } msat_model_iterator;
typedef int (*msat_solve_diversify_model_callback)(msat_model_iterator it,
                                                   void *user_data);
typedef int (*msat_ext_unsat_core_extractor)(int *cnf_in, int *groups_in_out,
                                             size_t *size_in_out,
                                             void *user_data);
void msat_free(void *ptr);
char *msat_get_version(void);
const char *msat_last_error_message(msat_env env);
msat_config msat_create_config(void);
msat_config msat_create_default_config(const char *logic);
msat_config msat_parse_config(const char *data);
msat_config msat_parse_config_file(FILE *f);
void msat_destroy_config(msat_config cfg);
msat_env msat_create_env(msat_config cfg);
msat_env msat_create_shared_env(msat_config cfg, msat_env sibling);
void msat_destroy_env(msat_env e);
int msat_set_option(msat_config cfg, const char *option, const char *value);

/* Types Declaration */
msat_type msat_get_bool_type(msat_env env);
msat_type msat_get_rational_type(msat_env env);
msat_type msat_get_integer_type(msat_env env);
msat_type msat_get_bv_type(msat_env env, size_t width);
msat_type msat_get_array_type(msat_env env, msat_type itp, msat_type etp);
msat_type msat_get_function_type(msat_env env, msat_type *param_types,
                                 size_t num_params, msat_type return_type);
/* Type Inspection */
int msat_is_bool_type(msat_env env, msat_type tp);
int msat_is_rational_type(msat_env env, msat_type tp);
int msat_is_integer_type(msat_env env, msat_type tp);
int msat_is_bv_type(msat_env env, msat_type tp, size_t *out_width);
int msat_is_array_type(msat_env env, msat_type tp,
                       msat_type *out_itp, msat_type *out_etp);

/* Term Declarations */
msat_decl msat_declare_function(msat_env e, const char *name, msat_type type);
msat_term msat_make_true(msat_env e);
msat_term msat_make_false(msat_env e);
msat_term msat_make_iff(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_or(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_and(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_not(msat_env e, msat_term t1);
msat_term msat_make_equal(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_eq(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_leq(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_plus(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_times(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_floor(msat_env e, msat_term t);
msat_term msat_make_number(msat_env e, const char *num_rep);
msat_term msat_make_term_ite(msat_env e, msat_term c,
                             msat_term tt, msat_term te);
msat_term msat_make_constant(msat_env e, msat_decl var);
msat_term msat_make_uf(msat_env e, msat_decl func, msat_term args[]);
msat_term msat_make_array_read(msat_env e, msat_term arr, msat_term idx);
msat_term msat_make_array_write(msat_env e, msat_term arr, msat_term idx,
                                msat_term elem);
msat_term msat_make_array_const(msat_env e, msat_type arrtp, msat_term elem);
msat_term msat_make_bv_number(msat_env e, const char *num_rep, size_t width,
                              size_t base);
msat_term msat_make_bv_concat(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_bv_extract(msat_env e, size_t msb, size_t lsb, msat_term t);
msat_term msat_make_bv_or(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_bv_xor(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_bv_and(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_bv_not(msat_env e, msat_term t);
msat_term msat_make_bv_lshl(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_bv_lshr(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_bv_ashr(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_bv_zext(msat_env e, size_t amount, msat_term t);
msat_term msat_make_bv_sext(msat_env e, size_t amount, msat_term t);
msat_term msat_make_bv_plus(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_bv_minus(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_bv_neg(msat_env e, msat_term t);
msat_term msat_make_bv_times(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_bv_udiv(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_bv_urem(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_bv_sdiv(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_bv_srem(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_bv_ult(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_bv_uleq(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_bv_slt(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_bv_sleq(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_bv_rol(msat_env e, size_t amount, msat_term t);
msat_term msat_make_bv_ror(msat_env e, size_t amount, msat_term t);
msat_term msat_make_bv_comp(msat_env e, msat_term t1, msat_term t2);
msat_term msat_make_int_to_bv(msat_env e, size_t width, msat_term t);
msat_term msat_make_int_from_ubv(msat_env e, msat_term t);
msat_term msat_make_int_from_sbv(msat_env e, msat_term t);
msat_term msat_make_term(msat_env e, msat_decl d, msat_term args[]);

/* Term Inspection */
size_t msat_term_id(msat_term t);
size_t msat_term_arity(msat_term t);
msat_term msat_term_get_arg(msat_term t, size_t n);
msat_type msat_term_get_type(msat_term t);
int msat_term_is_true(msat_env e, msat_term t);
int msat_term_is_false(msat_env e, msat_term t);
int msat_term_is_boolean_constant(msat_env e, msat_term t);
int msat_term_is_atom(msat_env e, msat_term t);
int msat_term_is_number(msat_env e, msat_term t);
int msat_term_is_and(msat_env e, msat_term t);
int msat_term_is_or(msat_env e, msat_term t);
int msat_term_is_not(msat_env e, msat_term t);
int msat_term_is_iff(msat_env e, msat_term t);
int msat_term_is_term_ite(msat_env e, msat_term t);
int msat_term_is_constant(msat_env e, msat_term t);
int msat_term_is_uf(msat_env e, msat_term t);
int msat_term_is_equal(msat_env e, msat_term t);
int msat_term_is_leq(msat_env e, msat_term t);
int msat_term_is_plus(msat_env e, msat_term t);
int msat_term_is_times(msat_env e, msat_term t);
int msat_term_is_floor(msat_env e, msat_term t);
int msat_term_is_array_read(msat_env e, msat_term t);
int msat_term_is_array_write(msat_env e, msat_term t);
int msat_term_is_array_const(msat_env e, msat_term t);
int msat_term_is_bv_concat(msat_env e, msat_term t);
int msat_term_is_bv_extract(msat_env e, msat_term t,
                            size_t *out_msb, size_t *out_lsb);
int msat_term_is_bv_or(msat_env e, msat_term t);
int msat_term_is_bv_xor(msat_env e, msat_term t);
int msat_term_is_bv_and(msat_env e, msat_term t);
int msat_term_is_bv_not(msat_env e, msat_term t);
int msat_term_is_bv_plus(msat_env e, msat_term t);
int msat_term_is_bv_minus(msat_env e, msat_term t);
int msat_term_is_bv_times(msat_env e, msat_term t);
int msat_term_is_bv_neg(msat_env e, msat_term t);
int msat_term_is_bv_udiv(msat_env e, msat_term t);
int msat_term_is_bv_urem(msat_env e, msat_term t);
int msat_term_is_bv_sdiv(msat_env e, msat_term t);
int msat_term_is_bv_srem(msat_env e, msat_term t);
int msat_term_is_bv_ult(msat_env e, msat_term t);
int msat_term_is_bv_uleq(msat_env e, msat_term t);
int msat_term_is_bv_slt(msat_env e, msat_term t);
int msat_term_is_bv_sleq(msat_env e, msat_term t);
int msat_term_is_bv_lshl(msat_env e, msat_term t);
int msat_term_is_bv_lshr(msat_env e, msat_term t);
int msat_term_is_bv_ashr(msat_env e, msat_term t);
int msat_term_is_bv_zext(msat_env e, msat_term t, size_t *out_amount);
int msat_term_is_bv_sext(msat_env e, msat_term t, size_t *out_amount);
int msat_term_is_bv_rol(msat_env e, msat_term t, size_t *out_amount);
int msat_term_is_bv_ror(msat_env e, msat_term t, size_t *out_amount);
int msat_term_is_bv_comp(msat_env e, msat_term t);
int msat_term_is_int_to_bv(msat_env e, msat_term t);
int msat_term_is_int_from_ubv(msat_env e, msat_term t);
int msat_term_is_int_from_sbv(msat_env e, msat_term t);
char *msat_term_repr(msat_term t);

/* Solver interaction */
int msat_push_backtrack_point(msat_env e);
int msat_pop_backtrack_point(msat_env e);
int msat_reset_env(msat_env e);
int msat_assert_formula(msat_env e, msat_term formula);
msat_result msat_solve(msat_env e);
msat_result msat_solve_with_assumptions(msat_env e, msat_term *assumptions,
                                        size_t num_assumptions);
int msat_all_sat(msat_env e, msat_term *important, size_t num_important,
                 msat_all_sat_model_callback func, void *user_data);
int msat_create_itp_group(msat_env e);
int msat_set_itp_group(msat_env e, int group);
msat_term msat_get_interpolant(msat_env e, int *groups_of_a, size_t n);
msat_term msat_get_model_value(msat_env e, msat_term term);
msat_model_iterator msat_create_model_iterator(msat_env e);
int msat_model_iterator_has_next(msat_model_iterator i);
int msat_model_iterator_next(msat_model_iterator i, msat_term *t, msat_term *v);
void msat_destroy_model_iterator(msat_model_iterator i);
msat_model msat_get_model(msat_env e);
void msat_destroy_model(msat_model m);
msat_term msat_model_eval(msat_model m, msat_term t);
msat_model_iterator msat_model_create_iterator(msat_model m);
msat_term *msat_get_unsat_core(msat_env e, size_t *core_size);
msat_term *msat_get_unsat_core_ext(msat_env e, size_t *core_size,
                                   msat_ext_unsat_core_extractor extractor,
                                   void *user_data);
msat_term *msat_get_unsat_assumptions(msat_env e, size_t *assumps_size);
"""

MSATEXIST_ELIM_MINIMAL_H ="""
typedef enum {
    MSAT_EXIST_ELIM_ALLSMT_FM,  /**< All-SMT and Fourier-Motzkin projection */
    MSAT_EXIST_ELIM_VTS         /**< Virtual term substitutions */
} msat_exist_elim_algorithm;

typedef struct {
    int toplevel_propagation; /**< enable toplevel propagation */
    int boolean_simplifications; /**< enable simplifications to the Boolean
                                  **< structure of the formula using
                                  **< And-Inverter Graph rewriting */
    int remove_redundant_constraints; /**< enable aggressive simplifications
                                       **< of redundant linear inequalities
                                       **< (can be very expensive) */
} msat_exist_elim_options;

msat_term msat_exist_elim(msat_env env, msat_term formula,
                          msat_term *vars_to_elim, size_t num_vars_to_elim,
                          msat_exist_elim_algorithm algo,
                          msat_exist_elim_options options);
"""

#
# Initialize the library. ffi and mathsat are the only two objects
# exported by the module
#
ffi, mathsat = init_lib()



if __name__ == "__main__":
    demo()

# NOTE: Most methods of the API are not being wrapped.  Some
#       automatization could be achieved by doing introspection of the
#       library, e.g.:
# for k in ffi._parser._declarations:
#     if k.startswith('function'):
#         print(k)
#         print(ffi._parser._declarations[k])
