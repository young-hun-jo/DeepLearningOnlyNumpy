is_simple_core = False

if is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import as_array, as_variable
    from dezero.core_simple import using_config, no_grad
    from dezero.core_simple import setup_variable
else:
    from dezero.core import Variable
    from dezero.core import as_array, as_variable
    from dezero.core import using_config, no_grad
    from dezero.core import setup_variable

setup_variable()