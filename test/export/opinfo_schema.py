# Owner(s): ["oncall: export"]

import torch
from torch._dispatch.python import enable_python_dispatcher
from torch._subclasses.schema_check_mode import SchemaCheckMode
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    ops,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import TestCase

test_classes = {}


class PreDispatchSchemaCheckMode(SchemaCheckMode):
    def __init__(self):
        self._dispatch_key = torch._C.DispatchKey.PreDispatch
        super().__init__()

    # creating this just so we have access to the offending op
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        try:
            return super().__torch_dispatch__(func, types, args=args, kwargs=kwargs)
        except RuntimeError as e:
            msg = e.args[0]
            e.args = (
                f"""SchemaCheckMode failed with the following error on op <{func}>, meaning
this op contains aliasing or mutations, despite claiming not to:\n\n"""
                + msg,
            )
            raise e


class TestOpInfo(TestCase):
    # @ops(op_db)
    @ops(op_db, allowed_dtypes=(torch.float, torch.int))
    def test_schema_check_op(self, device, dtype, op):
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)
        inputs = next(sample_inputs_itr)
        args = [inputs.input] + list(inputs.args)
        kwargs = inputs.kwargs
        with enable_python_dispatcher():
            with PreDispatchSchemaCheckMode():
                op.op(*args, **kwargs)


instantiate_device_type_tests(TestOpInfo, globals())

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
