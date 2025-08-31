# 🚀 Super Alita Automation - Quick Reference

## 🎯 Workflow Quick Actions

### 📦 Smart Release
```bash
Actions → Smart Release & Deployment → Run workflow
→ Select: patch/minor/major/prerelease → Run
```
**Use for**: Creating releases, deploying to staging

### 🔄 Update Dependencies
```bash
Actions → Intelligent Dependency Updates → Run workflow
→ Select: security/patch/minor/all → Select: basic/full/extended → Run
```
**Use for**: Security patches, dependency updates

### 📊 Performance Check
```bash
Actions → Performance Monitoring → Run workflow
→ Select: quick/full/stress/comparison → Run
```
**Use for**: Performance regression testing

### 🧠 Code Review
```bash
Actions → AI-Powered Code Review → Run workflow
→ Enter: PR number → Run
```
**Use for**: Manual code analysis of specific PRs

### 🏷️ Issue Management
```bash
Actions → Intelligent Issue & PR Management → Run workflow
```
**Use for**: Weekly maintenance, stale issue cleanup

### 🎛️ Health Check
```bash
Actions → Workflow Orchestrator → Run workflow
→ Select: health_check/emergency_stop/reset_automation/generate_report → Run
```
**Use for**: System monitoring, emergency controls

## 🏷️ Auto-Applied Labels

### Issues
- **Priority**: `priority: high/medium/low`
- **Type**: `type: bug/enhancement/question/documentation`
- **Component**: `component: deepcode/extension/wasm/core/automation`
- **Difficulty**: `difficulty: easy/medium/hard`
- **Special**: `good first issue`

### Pull Requests
- **Size**: `size: small/medium/large`
- **Type**: `type: feature/bugfix/documentation/refactoring/chore/tests`
- **Component**: `component: ci/cd/python/typescript/wasm/documentation/tests`
- **Status**: `status: work in progress/ready for review`
- **Special**: `breaking change`

## 🚨 Emergency Procedures

### Stop All Automation
```bash
Actions → Workflow Orchestrator → Run workflow → emergency_stop
```

### Health Dashboard
- **Location**: `docs/AUTOMATION_DASHBOARD.md`
- **Issue**: Search for label `automation-dashboard`

## 📈 Performance Thresholds

| Component | Threshold | Action |
|-----------|-----------|--------|
| Python functions | > 1 second | Flag as slow |
| Extension bundle | > 10 MB | Flag as large |
| WASM execution | > 1 second | Flag as slow |
| Memory usage | Track trends | Report regressions |

## 🔧 Common Commands

### Conventional Commits (for auto-versioning)
```bash
feat: add new feature          # Minor version bump
fix: resolve bug              # Patch version bump
feat!: breaking change        # Major version bump
docs: update documentation    # No version bump
chore: update dependencies    # No version bump
```

### Manual PR Review Trigger
```bash
# Comment on PR to trigger manual review
@github-actions run code-review
```

### Stale Issue Management
- **Auto-stale**: 30 days of inactivity
- **Auto-close**: 37 days (7 days after stale)
- **Exempt**: `pinned`, `priority: high`, `good first issue`

## 📊 Monitoring

### Health Status
- 🟢 **Healthy**: 80%+ success rate
- 🟡 **Warning**: 60-80% success rate
- 🔴 **Critical**: <60% success rate

### Key Metrics
- Workflow success rates
- Performance trends
- Security vulnerability counts
- Issue/PR velocity
- Code quality scores

## 🎯 Best Practices

### For Releases
1. Use conventional commit messages
2. Update changelogs manually if needed
3. Test staging deployment before production
4. Monitor release dashboard

### For Dependencies
1. Let automation handle security updates
2. Review minor/major updates manually
3. Check performance impact
4. Monitor for regressions

### For Code Quality
1. Address AI review suggestions
2. Keep PRs focused and small
3. Include tests for new features
4. Follow conventional commit format

### For Issues
1. Use descriptive titles
2. Include reproduction steps for bugs
3. Tag with appropriate components
4. Link related PRs with "Fixes #123"

---

*Quick reference for Super Alita's powerful automation workflows*
