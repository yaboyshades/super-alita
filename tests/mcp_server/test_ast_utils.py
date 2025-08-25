from mcp_server.ast_utils import rewrite_function_to_result


def test_rewrite_function_to_result_adds_return_and_diff():
    src = """
def target():
    result = 1 + 1
"""
    rewritten, diff = rewrite_function_to_result(src, "target")
    assert rewritten is not None
    assert "return result" in rewritten
    assert "+    return result" in diff
