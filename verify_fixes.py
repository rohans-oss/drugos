import os

checks = []

# 1. pytest.ini
with open('pytest.ini', encoding='utf-8', errors='replace') as f:
    content = f.read()
checks.append(('pytest.ini has CI marker filter', 'not network' in content or 'not slow' in content))

# 2. MANIFEST.in
with open('MANIFEST.in', encoding='utf-8', errors='replace') as f:
    content = f.read()
checks.append(('MANIFEST.in includes yaml/json', 'yaml' in content.lower()))

# 3. phase4/writeback.py
checks.append(('phase4/writeback.py exists', os.path.exists('phase4/writeback.py')))

# 4. scripts/restore_test.py
checks.append(('scripts/restore_test.py exists', os.path.exists('scripts/restore_test.py')))

# 5. CONTRIBUTING.md
checks.append(('CONTRIBUTING.md exists', os.path.exists('CONTRIBUTING.md')))

# 6. frontend /api/health route
checks.append(('frontend api/health route.ts exists', os.path.exists('frontend/src/app/api/health/route.ts')))

# 7. scripts/gt_api.py lifespan
if os.path.exists('scripts/gt_api.py'):
    with open('scripts/gt_api.py', encoding='utf-8', errors='replace') as f:
        gt = f.read()
    checks.append(('gt_api.py uses lifespan', 'lifespan' in gt))

# 8. phase1 /health
with open('phase1/service.py', encoding='utf-8', errors='replace') as f:
    p1 = f.read()
checks.append(('phase1/service.py has /health route', '/health' in p1))

# 9. .env in .gitignore
with open('.gitignore', encoding='utf-8', errors='replace') as f:
    gi = f.read()
checks.append(('.env in .gitignore', '.env' in gi))

# 10. slowapi in requirements.txt
with open('requirements.txt', encoding='utf-8', errors='replace') as f:
    req = f.read()
checks.append(('slowapi in requirements.txt', 'slowapi' in req))

# 11. phase1-service in docker-compose
with open('docker-compose.yml', encoding='utf-8', errors='replace') as f:
    dc = f.read()
checks.append(('phase1-service in docker-compose', 'phase1-service' in dc))

# 12. .env.example exists
checks.append(('.env.example exists', os.path.exists('.env.example')))

# 13. CI workflow exists
checks.append(('CI workflow exists', os.path.exists('.github/workflows/ci.yml')))

# 14. cross-phase test exists
checks.append(('cross-phase import test exists', os.path.exists('tests/test_cross_phase_imports.py')))

for name, ok in checks:
    print('OK    ' if ok else 'MISS  ', name)
