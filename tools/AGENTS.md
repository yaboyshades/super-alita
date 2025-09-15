# Tools & Utilities - Agent Instructions

## Overview
The `tools/` directory contains utility scripts and development tools for Super Alita:
- **Development Scripts** - Setup, deployment, and maintenance utilities
- **Data Processing** - Data migration and transformation tools
- **Analysis Tools** - System monitoring and diagnostic utilities
- **Build Tools** - Compilation and packaging scripts

## Directory Structure

### Tool Categories
```
tools/
├── setup/              # Setup and installation scripts
├── dev/                # Development utilities
├── build/              # Build and packaging tools
├── analysis/           # System analysis and monitoring
├── data/               # Data processing and migration
├── deployment/         # Deployment automation
└── maintenance/        # System maintenance scripts
```

## Development Tools

### Setup Scripts
```bash
# tools/setup/bootstrap.sh
#!/bin/bash
# Bootstrap development environment

set -e

echo "🚀 Bootstrapping Super Alita development environment..."

# Check prerequisites
python --version | grep -q "3.11" || {
    echo "❌ Python 3.11+ required"
    exit 1
}

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt -c constraints.txt
pip install -r requirements-test.txt

# Setup pre-commit hooks
echo "🪝 Setting up pre-commit hooks..."
pre-commit install

# Initialize configuration
echo "⚙️  Setting up configuration..."
cp .env.example .env
echo "Please edit .env with your configuration"

# Setup MCP server
echo "🔧 Setting up MCP server..."
pwsh ./Setup-MCP.ps1 -Bootstrap

echo "✅ Bootstrap complete! Run 'make test' to verify setup."
```

### Development Environment Manager
```python
# tools/dev/env_manager.py
"""Development environment management utilities"""

import subprocess
import sys
from pathlib import Path
from typing import List, Dict

class DevEnvironmentManager:
    """Manage development environment setup and validation"""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()

    def check_prerequisites(self) -> Dict[str, bool]:
        """Check if all prerequisites are installed"""
        checks = {
            "python_version": self._check_python_version(),
            "git": self._check_git(),
            "redis": self._check_redis(),
            "docker": self._check_docker(),
        }
        return checks

    def _check_python_version(self) -> bool:
        """Check Python version >= 3.11"""
        return sys.version_info >= (3, 11)

    def _check_git(self) -> bool:
        """Check if git is available"""
        try:
            subprocess.run(["git", "--version"],
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _check_redis(self) -> bool:
        """Check if Redis is available"""
        try:
            import redis
            client = redis.Redis(host='localhost', port=6379, socket_connect_timeout=1)
            client.ping()
            return True
        except Exception:
            return False

    def _check_docker(self) -> bool:
        """Check if Docker is available"""
        try:
            subprocess.run(["docker", "--version"],
                         capture_output=True, check=True)
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def setup_environment(self) -> bool:
        """Set up complete development environment"""
        steps = [
            ("Installing dependencies", self._install_dependencies),
            ("Setting up pre-commit", self._setup_precommit),
            ("Configuring environment", self._setup_config),
            ("Testing setup", self._test_setup),
        ]

        for description, step_func in steps:
            print(f"📋 {description}...")
            try:
                step_func()
                print(f"✅ {description} completed")
            except Exception as e:
                print(f"❌ {description} failed: {e}")
                return False

        return True

    def _install_dependencies(self):
        """Install project dependencies"""
        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "-r", "requirements.txt",
            "-c", "constraints.txt"
        ], check=True)

        subprocess.run([
            sys.executable, "-m", "pip", "install",
            "-r", "requirements-test.txt"
        ], check=True)

    def _setup_precommit(self):
        """Set up pre-commit hooks"""
        subprocess.run(["pre-commit", "install"], check=True)

    def _setup_config(self):
        """Set up configuration files"""
        env_example = self.project_root / ".env.example"
        env_file = self.project_root / ".env"

        if env_example.exists() and not env_file.exists():
            env_file.write_text(env_example.read_text())

    def _test_setup(self):
        """Test the setup"""
        subprocess.run([
            sys.executable, "-m", "pytest",
            "tests/unit/", "-q", "--tb=short"
        ], check=True)

if __name__ == "__main__":
    manager = DevEnvironmentManager()

    print("🔍 Checking prerequisites...")
    checks = manager.check_prerequisites()

    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")

    if all(checks.values()):
        print("\n🚀 Setting up environment...")
        success = manager.setup_environment()
        if success:
            print("\n🎉 Development environment ready!")
        else:
            print("\n💥 Setup failed!")
            sys.exit(1)
    else:
        print("\n💥 Prerequisites not met!")
        sys.exit(1)
```

## Build Tools

### Project Builder
```python
# tools/build/builder.py
"""Build and packaging tools for Super Alita"""

import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import List, Optional

class ProjectBuilder:
    """Build and package Super Alita for deployment"""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()
        self.build_dir = self.project_root / "build"
        self.dist_dir = self.project_root / "dist"

    def clean(self):
        """Clean build artifacts"""
        for directory in [self.build_dir, self.dist_dir]:
            if directory.exists():
                shutil.rmtree(directory)

        # Remove Python cache
        for cache_dir in self.project_root.rglob("__pycache__"):
            shutil.rmtree(cache_dir)

        # Remove .pyc files
        for pyc_file in self.project_root.rglob("*.pyc"):
            pyc_file.unlink()

    def build_source_distribution(self) -> Path:
        """Build source distribution"""
        self.build_dir.mkdir(exist_ok=True)

        # Copy source files
        source_files = [
            "src/",
            "mcp_server/",
            "requirements.txt",
            "constraints.txt",
            "pyproject.toml",
            "README.md",
            "LICENSE",
        ]

        build_root = self.build_dir / "super-alita"
        build_root.mkdir(exist_ok=True)

        for item in source_files:
            source_path = self.project_root / item
            if source_path.exists():
                if source_path.is_dir():
                    shutil.copytree(source_path, build_root / item)
                else:
                    shutil.copy2(source_path, build_root / item)

        # Create archive
        self.dist_dir.mkdir(exist_ok=True)
        archive_path = self.dist_dir / "super-alita-source.zip"

        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for file_path in build_root.rglob("*"):
                if file_path.is_file():
                    arcname = file_path.relative_to(self.build_dir)
                    zipf.write(file_path, arcname)

        return archive_path

    def build_wheel(self) -> Path:
        """Build Python wheel"""
        subprocess.run([
            "python", "-m", "build",
            "--wheel",
            "--outdir", str(self.dist_dir)
        ], check=True, cwd=self.project_root)

        # Find the created wheel
        wheel_files = list(self.dist_dir.glob("*.whl"))
        return wheel_files[-1] if wheel_files else None

    def build_docker_image(self, tag: str = "super-alita:latest") -> str:
        """Build Docker image"""
        subprocess.run([
            "docker", "build",
            "-t", tag,
            "."
        ], check=True, cwd=self.project_root)

        return tag

    def run_tests_in_build(self) -> bool:
        """Run tests against built package"""
        try:
            subprocess.run([
                "python", "-m", "pytest",
                "tests/", "-v",
                "--tb=short"
            ], check=True, cwd=self.build_dir / "super-alita")
            return True
        except subprocess.CalledProcessError:
            return False

def main():
    """Main build script"""
    import argparse

    parser = argparse.ArgumentParser(description="Build Super Alita")
    parser.add_argument("--clean", action="store_true",
                       help="Clean build artifacts")
    parser.add_argument("--source", action="store_true",
                       help="Build source distribution")
    parser.add_argument("--wheel", action="store_true",
                       help="Build wheel")
    parser.add_argument("--docker", action="store_true",
                       help="Build Docker image")
    parser.add_argument("--test", action="store_true",
                       help="Run tests after build")
    parser.add_argument("--all", action="store_true",
                       help="Build everything")

    args = parser.parse_args()

    builder = ProjectBuilder()

    if args.clean or args.all:
        print("🧹 Cleaning build artifacts...")
        builder.clean()

    if args.source or args.all:
        print("📦 Building source distribution...")
        archive_path = builder.build_source_distribution()
        print(f"✅ Source distribution: {archive_path}")

    if args.wheel or args.all:
        print("🎡 Building wheel...")
        wheel_path = builder.build_wheel()
        print(f"✅ Wheel: {wheel_path}")

    if args.docker or args.all:
        print("🐳 Building Docker image...")
        image_tag = builder.build_docker_image()
        print(f"✅ Docker image: {image_tag}")

    if args.test or args.all:
        print("🧪 Running tests...")
        if builder.run_tests_in_build():
            print("✅ Tests passed")
        else:
            print("❌ Tests failed")

if __name__ == "__main__":
    main()
```

## Analysis Tools

### System Monitor
```python
# tools/analysis/system_monitor.py
"""System monitoring and analysis tools"""

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any
import psutil
import aiohttp

class SystemMonitor:
    """Monitor Super Alita system health and performance"""

    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.metrics = []

    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics"""
        process = psutil.Process()

        metrics = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "cpu": {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
            },
            "memory": {
                "total": psutil.virtual_memory().total,
                "available": psutil.virtual_memory().available,
                "percent": psutil.virtual_memory().percent,
                "process_rss": process.memory_info().rss,
                "process_vms": process.memory_info().vms
            },
            "disk": {
                "total": psutil.disk_usage('/').total,
                "free": psutil.disk_usage('/').free,
                "percent": psutil.disk_usage('/').percent
            },
            "network": {
                "bytes_sent": psutil.net_io_counters().bytes_sent,
                "bytes_recv": psutil.net_io_counters().bytes_recv,
                "packets_sent": psutil.net_io_counters().packets_sent,
                "packets_recv": psutil.net_io_counters().packets_recv
            }
        }

        return metrics

    async def check_application_health(self, base_url: str = "http://localhost:8080") -> Dict[str, Any]:
        """Check application health endpoints"""
        health_checks = {
            "main_health": f"{base_url}/health",
            "runtime_health": f"{base_url}/healthz",
            "mcp_health": f"{base_url}/mcp/health"
        }

        results = {}

        async with aiohttp.ClientSession() as session:
            for check_name, url in health_checks.items():
                try:
                    start_time = time.time()
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        response_time = time.time() - start_time

                        results[check_name] = {
                            "status": "healthy" if response.status == 200 else "unhealthy",
                            "status_code": response.status,
                            "response_time": response_time,
                            "timestamp": datetime.now(timezone.utc).isoformat()
                        }

                        if response.status == 200:
                            try:
                                body = await response.json()
                                results[check_name]["details"] = body
                            except:
                                results[check_name]["details"] = await response.text()

                except Exception as e:
                    results[check_name] = {
                        "status": "error",
                        "error": str(e),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    }

        return results

    async def analyze_event_bus_metrics(self) -> Dict[str, Any]:
        """Analyze event bus performance"""
        try:
            import redis
            r = redis.Redis(host='localhost', port=6379)

            info = r.info()

            metrics = {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "instantaneous_ops_per_sec": info.get("instantaneous_ops_per_sec", 0),
                "keyspace": {}
            }

            # Get keyspace info
            for key, value in info.items():
                if key.startswith("db"):
                    metrics["keyspace"][key] = value

            return metrics

        except Exception as e:
            return {"error": str(e), "available": False}

    async def generate_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive health report"""
        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "overall_status": "unknown",
            "system_metrics": await self.collect_system_metrics(),
            "application_health": await self.check_application_health(),
            "event_bus_metrics": await self.analyze_event_bus_metrics(),
        }

        # Determine overall status
        app_health = report["application_health"]
        healthy_services = sum(1 for service in app_health.values()
                             if service.get("status") == "healthy")
        total_services = len(app_health)

        if healthy_services == total_services:
            report["overall_status"] = "healthy"
        elif healthy_services > 0:
            report["overall_status"] = "degraded"
        else:
            report["overall_status"] = "unhealthy"

        return report

    async def continuous_monitoring(self, interval: int = 60, output_file: str = None):
        """Run continuous monitoring"""
        output_path = Path(output_file) if output_file else Path("monitoring.log")

        print(f"🔍 Starting continuous monitoring (interval: {interval}s)")
        print(f"📝 Writing to: {output_path}")

        try:
            while True:
                report = await self.generate_health_report()

                # Log to file
                with open(output_path, "a") as f:
                    f.write(json.dumps(report) + "\n")

                # Print status
                status_emoji = {
                    "healthy": "✅",
                    "degraded": "⚠️",
                    "unhealthy": "❌",
                    "unknown": "❓"
                }

                emoji = status_emoji.get(report["overall_status"], "❓")
                timestamp = report["timestamp"]
                status = report["overall_status"]

                print(f"{emoji} {timestamp} - Status: {status}")

                await asyncio.sleep(interval)

        except KeyboardInterrupt:
            print("\n🛑 Monitoring stopped")

async def main():
    """Main monitoring script"""
    import argparse

    parser = argparse.ArgumentParser(description="Super Alita System Monitor")
    parser.add_argument("--report", action="store_true",
                       help="Generate single health report")
    parser.add_argument("--monitor", action="store_true",
                       help="Run continuous monitoring")
    parser.add_argument("--interval", type=int, default=60,
                       help="Monitoring interval in seconds")
    parser.add_argument("--output", type=str,
                       help="Output file for monitoring logs")

    args = parser.parse_args()

    monitor = SystemMonitor()

    if args.report:
        print("📊 Generating health report...")
        report = await monitor.generate_health_report()
        print(json.dumps(report, indent=2))

    elif args.monitor:
        await monitor.continuous_monitoring(
            interval=args.interval,
            output_file=args.output
        )

    else:
        print("Use --report for single report or --monitor for continuous monitoring")

if __name__ == "__main__":
    asyncio.run(main())
```

## Data Processing Tools

### Data Migration Utility
```python
# tools/data/migration.py
"""Data migration and transformation utilities"""

import json
import asyncio
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone

class DataMigrator:
    """Handle data migration between versions"""

    def __init__(self, source_dir: Path, target_dir: Path):
        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)

    async def migrate_neural_atoms(self, version_from: str, version_to: str) -> bool:
        """Migrate neural atoms between versions"""

        print(f"🔄 Migrating neural atoms from {version_from} to {version_to}")

        # Define migration strategies
        migrations = {
            ("1.0", "2.0"): self._migrate_v1_to_v2,
            ("2.0", "3.0"): self._migrate_v2_to_v3,
        }

        migration_func = migrations.get((version_from, version_to))
        if not migration_func:
            print(f"❌ No migration path from {version_from} to {version_to}")
            return False

        try:
            await migration_func()
            print("✅ Migration completed successfully")
            return True
        except Exception as e:
            print(f"❌ Migration failed: {e}")
            return False

    async def _migrate_v1_to_v2(self):
        """Migrate from version 1.0 to 2.0 format"""

        # v1.0 format: simple JSON files
        # v2.0 format: structured atoms with metadata

        source_files = list(self.source_dir.glob("*.json"))

        for source_file in source_files:
            print(f"📄 Processing {source_file.name}")

            with open(source_file) as f:
                v1_data = json.load(f)

            # Transform to v2.0 format
            v2_atom = {
                "uuid": self._generate_uuid_v2(v1_data),
                "content": v1_data,
                "atom_type": "migrated_data",
                "title": f"Migrated from {source_file.name}",
                "metadata": {
                    "migration": {
                        "source_version": "1.0",
                        "target_version": "2.0",
                        "migration_date": datetime.now(timezone.utc).isoformat(),
                        "source_file": str(source_file)
                    }
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

            # Write to target
            target_file = self.target_dir / f"{v2_atom['uuid']}.json"
            self.target_dir.mkdir(parents=True, exist_ok=True)

            with open(target_file, 'w') as f:
                json.dump(v2_atom, f, indent=2)

    async def _migrate_v2_to_v3(self):
        """Migrate from version 2.0 to 3.0 format"""

        # v2.0 format: structured atoms
        # v3.0 format: atoms with bonds and enhanced metadata

        source_files = list(self.source_dir.glob("*.json"))
        atoms_with_bonds = []

        for source_file in source_files:
            with open(source_file) as f:
                v2_atom = json.load(f)

            # Transform to v3.0 format
            v3_atom = {
                **v2_atom,
                "bonds": [],  # Initialize empty bonds
                "metadata": {
                    **v2_atom.get("metadata", {}),
                    "version": "3.0",
                    "enhanced_features": True
                }
            }

            atoms_with_bonds.append(v3_atom)

        # Create semantic bonds between related atoms
        for i, atom1 in enumerate(atoms_with_bonds):
            for j, atom2 in enumerate(atoms_with_bonds[i+1:], i+1):
                if self._atoms_are_related(atom1, atom2):
                    bond = {
                        "source_uuid": atom1["uuid"],
                        "target_uuid": atom2["uuid"],
                        "bond_type": "semantic",
                        "strength": 0.7,
                        "metadata": {
                            "created_during_migration": True
                        }
                    }
                    atom1["bonds"].append(bond)

        # Write migrated atoms
        self.target_dir.mkdir(parents=True, exist_ok=True)
        for atom in atoms_with_bonds:
            target_file = self.target_dir / f"{atom['uuid']}.json"
            with open(target_file, 'w') as f:
                json.dump(atom, f, indent=2)

    def _generate_uuid_v2(self, data: Dict) -> str:
        """Generate UUID for v2.0 format"""
        import hashlib
        import uuid

        content_str = json.dumps(data, sort_keys=True)
        content_hash = hashlib.md5(content_str.encode()).hexdigest()
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, content_hash))

    def _atoms_are_related(self, atom1: Dict, atom2: Dict) -> bool:
        """Determine if two atoms are semantically related"""

        # Simple heuristic: check for common keywords
        content1 = str(atom1.get("content", "")).lower()
        content2 = str(atom2.get("content", "")).lower()

        # Extract keywords
        words1 = set(content1.split())
        words2 = set(content2.split())

        # Calculate overlap
        overlap = len(words1 & words2)
        union = len(words1 | words2)

        if union == 0:
            return False

        similarity = overlap / union
        return similarity > 0.3  # Threshold for relatedness

async def main():
    """Main migration script"""
    import argparse

    parser = argparse.ArgumentParser(description="Data Migration Tool")
    parser.add_argument("--source", required=True, help="Source directory")
    parser.add_argument("--target", required=True, help="Target directory")
    parser.add_argument("--from-version", required=True, help="Source version")
    parser.add_argument("--to-version", required=True, help="Target version")

    args = parser.parse_args()

    migrator = DataMigrator(args.source, args.target)

    success = await migrator.migrate_neural_atoms(
        args.from_version,
        args.to_version
    )

    if success:
        print("🎉 Migration completed successfully!")
    else:
        print("💥 Migration failed!")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
```

## Deployment Tools

### Deployment Automation
```bash
# tools/deployment/deploy.sh
#!/bin/bash
# Deployment automation script

set -e

ENVIRONMENT=${1:-staging}
VERSION=${2:-latest}

echo "🚀 Deploying Super Alita to $ENVIRONMENT (version: $VERSION)"

# Validate environment
case $ENVIRONMENT in
    staging|production)
        echo "✅ Valid environment: $ENVIRONMENT"
        ;;
    *)
        echo "❌ Invalid environment: $ENVIRONMENT"
        echo "Usage: deploy.sh <staging|production> [version]"
        exit 1
        ;;
esac

# Load environment-specific configuration
CONFIG_FILE="tools/deployment/config/$ENVIRONMENT.env"
if [ -f "$CONFIG_FILE" ]; then
    echo "📋 Loading configuration: $CONFIG_FILE"
    source "$CONFIG_FILE"
else
    echo "❌ Configuration file not found: $CONFIG_FILE"
    exit 1
fi

# Pre-deployment checks
echo "🔍 Running pre-deployment checks..."

# Check if target servers are accessible
for SERVER in $DEPLOY_SERVERS; do
    echo "  📡 Checking $SERVER..."
    if ! ssh -o ConnectTimeout=10 "$SERVER" "echo 'Connection successful'" > /dev/null 2>&1; then
        echo "❌ Cannot connect to $SERVER"
        exit 1
    fi
done

# Build and test
echo "🔨 Building application..."
python tools/build/builder.py --all --test

# Deploy to servers
for SERVER in $DEPLOY_SERVERS; do
    echo "📦 Deploying to $SERVER..."

    # Upload files
    scp dist/super-alita-source.zip "$SERVER:/tmp/"

    # Execute deployment on remote server
    ssh "$SERVER" << EOF
        set -e
        cd $DEPLOY_PATH

        # Backup current version
        if [ -d "super-alita" ]; then
            mv super-alita super-alita.backup.\$(date +%Y%m%d_%H%M%S)
        fi

        # Extract new version
        unzip -q /tmp/super-alita-source.zip
        cd super-alita

        # Install dependencies
        pip install -r requirements.txt -c constraints.txt

        # Run database migrations if needed
        if [ -f "tools/data/migration.py" ]; then
            python tools/data/migration.py --auto-migrate
        fi

        # Start services
        systemctl restart super-alita
        systemctl restart super-alita-mcp

        # Health check
        sleep 10
        curl -f http://localhost:8080/health || exit 1

        echo "✅ Deployment successful on $SERVER"
EOF
done

echo "🎉 Deployment to $ENVIRONMENT completed successfully!"

# Post-deployment verification
echo "🧪 Running post-deployment tests..."
python tools/deployment/verify_deployment.py --environment "$ENVIRONMENT"

echo "✅ All done!"
```

## Maintenance Tools

### System Cleanup
```python
# tools/maintenance/cleanup.py
"""System maintenance and cleanup utilities"""

import shutil
import asyncio
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import List, Dict

class SystemCleaner:
    """Clean up system artifacts and temporary files"""

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path.cwd()

    def clean_build_artifacts(self) -> Dict[str, int]:
        """Clean build artifacts and cache files"""
        cleaned = {"files": 0, "directories": 0, "bytes": 0}

        patterns_to_clean = [
            "**/__pycache__",
            "**/*.pyc",
            "**/*.pyo",
            "**/*.pyd",
            "**/build",
            "**/dist",
            "**/.pytest_cache",
            "**/.coverage",
            "**/htmlcov",
            "**/*.egg-info",
        ]

        for pattern in patterns_to_clean:
            for path in self.project_root.glob(pattern):
                if path.is_file():
                    size = path.stat().st_size
                    path.unlink()
                    cleaned["files"] += 1
                    cleaned["bytes"] += size
                elif path.is_dir():
                    size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
                    shutil.rmtree(path)
                    cleaned["directories"] += 1
                    cleaned["bytes"] += size

        return cleaned

    def clean_logs(self, days_to_keep: int = 30) -> Dict[str, int]:
        """Clean old log files"""
        cleaned = {"files": 0, "bytes": 0}
        cutoff_date = datetime.now() - timedelta(days=days_to_keep)

        log_dirs = [
            self.project_root / "logs",
            self.project_root / "data" / "logs",
            Path("/tmp") / "super-alita-logs"
        ]

        for log_dir in log_dirs:
            if not log_dir.exists():
                continue

            for log_file in log_dir.glob("*.log*"):
                if log_file.is_file():
                    file_time = datetime.fromtimestamp(log_file.stat().st_mtime)
                    if file_time < cutoff_date:
                        size = log_file.stat().st_size
                        log_file.unlink()
                        cleaned["files"] += 1
                        cleaned["bytes"] += size

        return cleaned

    async def clean_neural_store(self, days_to_keep: int = 90) -> Dict[str, int]:
        """Clean old neural atoms and bonds"""
        # This would integrate with the neural store to clean old data
        # Implementation depends on neural store backend

        cleaned = {"atoms": 0, "bonds": 0}

        try:
            from src.neural.store import NeuralStore

            store = NeuralStore()
            cutoff_date = datetime.now(timezone.utc) - timedelta(days=days_to_keep)

            # Clean old atoms (if store supports it)
            if hasattr(store, 'cleanup_old_atoms'):
                result = await store.cleanup_old_atoms(cutoff_date)
                cleaned.update(result)

        except ImportError:
            print("⚠️  Neural store not available for cleanup")

        return cleaned

    def generate_cleanup_report(self) -> Dict[str, any]:
        """Generate comprehensive cleanup report"""

        print("🧹 Starting system cleanup...")

        # Clean build artifacts
        print("  🔨 Cleaning build artifacts...")
        build_cleaned = self.clean_build_artifacts()

        # Clean logs
        print("  📋 Cleaning old logs...")
        logs_cleaned = self.clean_logs()

        # Calculate totals
        total_files = build_cleaned["files"] + logs_cleaned["files"]
        total_bytes = build_cleaned["bytes"] + logs_cleaned["bytes"]

        report = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "build_artifacts": build_cleaned,
            "logs": logs_cleaned,
            "totals": {
                "files_removed": total_files,
                "bytes_freed": total_bytes,
                "mb_freed": round(total_bytes / (1024 * 1024), 2)
            }
        }

        return report

async def main():
    """Main cleanup script"""
    import argparse

    parser = argparse.ArgumentParser(description="System Cleanup Tool")
    parser.add_argument("--build", action="store_true",
                       help="Clean build artifacts")
    parser.add_argument("--logs", action="store_true",
                       help="Clean old logs")
    parser.add_argument("--neural", action="store_true",
                       help="Clean neural store")
    parser.add_argument("--all", action="store_true",
                       help="Clean everything")
    parser.add_argument("--log-days", type=int, default=30,
                       help="Days of logs to keep")
    parser.add_argument("--neural-days", type=int, default=90,
                       help="Days of neural data to keep")

    args = parser.parse_args()

    cleaner = SystemCleaner()

    if args.all or not any([args.build, args.logs, args.neural]):
        # Clean everything
        report = cleaner.generate_cleanup_report()

        if args.neural or args.all:
            neural_cleaned = await cleaner.clean_neural_store(args.neural_days)
            report["neural_store"] = neural_cleaned

        # Print summary
        print("\n📊 Cleanup Summary:")
        print(f"  📁 Files removed: {report['totals']['files_removed']}")
        print(f"  💾 Space freed: {report['totals']['mb_freed']} MB")
        print("✅ Cleanup completed!")

    else:
        # Clean specific components
        if args.build:
            result = cleaner.clean_build_artifacts()
            print(f"🔨 Build cleanup: {result['files']} files, {result['directories']} dirs")

        if args.logs:
            result = cleaner.clean_logs(args.log_days)
            print(f"📋 Log cleanup: {result['files']} files")

        if args.neural:
            result = await cleaner.clean_neural_store(args.neural_days)
            print(f"🧠 Neural cleanup: {result['atoms']} atoms, {result['bonds']} bonds")

if __name__ == "__main__":
    asyncio.run(main())
```

## Tool Usage Guidelines

### Running Tools
```bash
# Setup and development
./tools/setup/bootstrap.sh                    # Bootstrap environment
python tools/dev/env_manager.py               # Check/setup environment

# Repository chunk manifests
python tools/chunk_repo.py                    # Generate chunks/chunk manifests
python tools/chunk_repo.py --dry-run          # Preview without writing files
python tools/chunk_repo.py --output-dir tmp   # Customise manifest directory

# Building and packaging
python tools/build/builder.py --all           # Build everything
python tools/build/builder.py --wheel         # Build wheel only

# Monitoring and analysis
python tools/analysis/system_monitor.py --report    # Single health report
python tools/analysis/system_monitor.py --monitor   # Continuous monitoring

# Data management
python tools/data/migration.py --source old --target new --from-version 1.0 --to-version 2.0

# Deployment
./tools/deployment/deploy.sh staging          # Deploy to staging
./tools/deployment/deploy.sh production v1.2  # Deploy specific version

# Maintenance
python tools/maintenance/cleanup.py --all     # Full system cleanup
python tools/maintenance/cleanup.py --logs    # Clean logs only
```

### Chunk Manifest Generator
- **Purpose** - Emits manifests of Python files grouped by top-level directory
  while respecting the scope boundaries defined by `AGENTS.md` files.
- **Output** - Creates (or updates) text files in `chunks/` (e.g.,
  `chunks/core.txt` or `chunks/tests-runtime.txt`) listing repository-relative
  paths.
- **Options** - Use `--dry-run` to inspect the manifests without writing them
  and `--output-dir` to target an alternate directory for the generated files.


### Best Practices

#### Tool Development
- **Single responsibility** - Each tool should have one clear purpose
- **Configuration** - Use configuration files for environment-specific settings
- **Error handling** - Provide clear error messages and exit codes
- **Documentation** - Include help text and usage examples
- **Testing** - Test tools in isolated environments

#### Security
- **Credential management** - Never hardcode secrets in tools
- **Input validation** - Validate all user inputs
- **Safe operations** - Use dry-run modes for destructive operations
- **Audit logging** - Log all tool executions for auditing

#### Performance
- **Async operations** - Use async for I/O-heavy operations
- **Progress reporting** - Show progress for long-running operations
- **Resource limits** - Implement timeouts and resource limits
- **Cleanup** - Always clean up resources and temporary files
