is_simple_core = False

if is_simple_core:
    from dezero.core_simple import Variable, Function
    from dezero.core_simple import using_config, no_grad
    from dezero.core_simple import as_array, as_variable
    from dezero.core_simple import setup_variable

else:
    from dezero.core import Variable, Function
    from dezero.core import using_config, no_grad
    from dezero.core import as_array, as_variable
    from dezero.core import setup_variable
    from dezero.core import Parameter
    from dezero.layers import Layer
    from dezero.dataloaders import DataLoader
    from dezero.core import Config
    from dezero.core import test_mode

setup_variable()