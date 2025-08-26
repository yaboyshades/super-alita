from __future__ import annotations

from difflib import unified_diff
import libcst as cst


class _ReturnResultTransformer(cst.CSTTransformer):
    def __init__(self, function_name: str) -> None:
        self.function_name = function_name
        self.found = False
        self.result_var: str | None = None

    def leave_FunctionDef(
        self, original_node: cst.FunctionDef, updated_node: cst.FunctionDef
    ) -> cst.FunctionDef:
        if original_node.name.value != self.function_name:
            return updated_node

        self.found = True
        body_stmts = list(updated_node.body.body)
        if body_stmts and isinstance(
            body_stmts[-1], cst.SimpleStatementLine
        ) and any(isinstance(s, cst.Return) for s in body_stmts[-1].body):
            return updated_node

        last_assign: str | None = None
        for stmt in body_stmts:
            if isinstance(stmt, cst.SimpleStatementLine):
                for small in stmt.body:
                    if (
                        isinstance(small, cst.Assign)
                        and isinstance(small.targets[0].target, cst.Name)
                    ):
                        last_assign = small.targets[0].target.value

        if last_assign is None:
            return updated_node

        self.result_var = last_assign
        return_stmt = cst.SimpleStatementLine(
            [cst.Return(cst.Name(last_assign))]
        )
        new_body = body_stmts + [return_stmt]
        return updated_node.with_changes(
            body=updated_node.body.with_changes(body=new_body)
        )


def rewrite_function_to_result(
    src: str, function_name: str
) -> tuple[str | None, str | None]:
    try:
        module = cst.parse_module(src)
    except Exception as e:
        return None, str(e)

    transformer = _ReturnResultTransformer(function_name)
    new_module = module.visit(transformer)

    if not transformer.found:
        return None, "Function not found"
    if transformer.result_var is None:
        return None, "Could not determine result variable"

    new_src = new_module.code
    diff = "".join(
        unified_diff(
            src.splitlines(keepends=True),
            new_src.splitlines(keepends=True),
            fromfile="a.py",
            tofile="b.py",
        )
    )
    return new_src, diff
