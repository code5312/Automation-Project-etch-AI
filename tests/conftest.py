"""테스트 전역 설정. etchpipe.serve.app은 import 시점에 ETCH_API_KEY를 요구하므로
테스트 수집 전에 미리 넣어둔다(실제 운영 키가 아니라 테스트 전용 값)."""

import os

os.environ.setdefault("ETCH_API_KEY", "test-key-not-for-production")
