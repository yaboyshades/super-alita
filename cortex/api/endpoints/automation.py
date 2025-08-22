from fastapi import APIRouter, Body

from cortex.automation.deployment import DeploymentAutomation
from cortex.automation.git_workflow import GitAutomation
from cortex.automation.python_tests import PythonTestAutomation
from cortex.automation.typescript_tests import ExtensionTestAutomation

router = APIRouter(prefix="/api/automation", tags=["automation"])


@router.post("/run-python-tests")
async def run_python_tests():
    return PythonTestAutomation().run_pytest_with_coverage()


@router.post("/format-and-lint")
async def format_and_lint():
    return PythonTestAutomation().format_and_lint()


@router.post("/run-extension-tests")
async def run_extension_tests():
    return ExtensionTestAutomation().run_extension_tests()


@router.post("/build-extension")
async def build_extension():
    return ExtensionTestAutomation().build_extension()


@router.post("/create-feature-branch/{feature_name}")
async def create_feature_branch(feature_name: str):
    return GitAutomation().create_feature_branch(feature_name)


@router.post("/auto-commit")
async def auto_commit(
    message: str = Body(..., embed=True),
    files: list[str] | None = Body(default=None, embed=True),  # noqa: B008
):
    return GitAutomation().auto_commit(message, files)


@router.post("/deploy-test")
async def deploy_test():
    return DeploymentAutomation().deploy_to_test()


@router.post("/package-extension")
async def package_extension():
    return DeploymentAutomation().package_extension()
