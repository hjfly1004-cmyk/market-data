# market-data

미국장 마감 후 매일 자동으로 시세를 받아 저장하고, 토니크로스 매매기준 지표를 계산합니다.

- 수집 종목: VIX, VIX3M, VXX, SPX(^GSPC), SPY, QQQ, SOXX, SOXL
- 실행: 매일 한국시간 06:30 (GitHub Actions `시세 수집`), Actions 탭에서 수동 실행도 가능
- 결과
  - `data/<종목>.csv`: 최근 5년 일봉
  - `data/summary.json`: 시장 국면(VIX 주봉 스토캐스틱, VXX DMI, VIX/VIX3M)과 종목별 MACD(19,39,9)·이평·주봉 MFI 등

주봉 값은 마감된 주봉 기준이며, 진행 중인 주의 값은 `_진행중` 항목으로 따로 기록합니다.
