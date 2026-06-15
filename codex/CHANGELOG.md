# Changelog

鏈枃浠惰褰?`codex/` 瀹炵幇闈㈠悜浣跨敤鑰呯殑绋冲畾鐗堟湰鍙樺寲銆?

## [2.0.0] - 2026-06-16

### Added

- 新增仅含 Codex 实现的课程报告资料包指南与可重复打包脚本。
- 新增完整 43 类高级训练管线与 LeNet/Enhanced 架构选择。
- 新增 Momentum SGD、动态增强、分层 track 验证集、学习率调度和最佳 checkpoint。
- 模型格式升级至 v2 以记录网络架构，同时兼容既有 v1 LeNet 权重。
- LibTorch GPU 加速后端：新增 CPPCNN_WITH_LIBTORCH 编译选项。
- GPU 加速训练 CLI：cppcnn_app_gpu 可执行文件，支持 train/evaluate/predict。
- LibTorch LeNet 与 Enhanced 模型定义，自动推导 CUDA 架构。
- GPU 训练循环：batch forward/backward、SGD with momentum、CUDA 同步。
- 完整训练指标：train/val loss、accuracy、mean class accuracy、min class accuracy。
- 与原始手写 CNN 实现完全独立，互不干扰。

### Changed

- CMakeLists.txt：新增 CPPCNN_WITH_LIBTORCH 选项。
- 新增 codex/src/cnn_libtorch/ 目录，包含 LibTorch 模型、数据集、训练器。
- 训练速度对比：1 epoch LeNet(CPU) = 32.8s → 1 epoch LeNet(GPU) = 2.2s，加速约 15 倍。

### Known Limitations

- GPU 验证阶段的前向传播存在 CUDA 兼容性问题，当前验证和评估在 CPU 上执行。
- 需要 NVIDIA GPU、CUDA 13.0+ 和 LibTorch 2.12.0-cu130。

## 1.1.0 - 2026-06-13

- 鏂板璇箟鍧囪　 10 绫昏缁冨伐浣滄祦鍜岀嫭绔嬫ā鍨嬨€?
- 寮€鍙戣€呰缁冭剼鏈敮鎸佸懡鍚嶅弬鏁般€佽嚜鍔ㄨ瘎浼般€佹棩蹇椼€佸厓鏁版嵁鍜屽巻鍙叉ā鍨嬪綊妗ｃ€?
- CLI 璁粌鍏ュ彛寮€鏀?batch size銆佸涔犵巼銆佹潈閲嶈“鍑忓拰闅忔満绉嶅瓙銆?
- 瀛愰泦宸ュ叿鏀寔鏄惧紡 GTSRB 绫诲埆 ID銆?
- GUI 鍔犺浇妯″瀷鏃朵紭鍏堝尮閰嶅悓鍚?`.labels.txt`锛屾敮鎸佸畨鍏ㄥ垏鎹笉鍚岀被鍒槧灏勩€?
- 浠撳簱鍚堝叆 Claude Python/PyTorch 瀵圭収鍘熷瀷锛屽苟鏂板鐙珛 CI 涓庨」鐩竟鐣岃鏄庛€?

## 1.0.1 - 2026-06-13

### Changed

- 婕旂ず鍖轰粠浜斿紶杩炵画闄愰€熺墝鏀逛负浜斾釜涓嶅悓妯″瀷绫诲埆銆?
- 鏂版紨绀虹粍鍚堣鐩?30 km/h銆?00 km/h銆佺姝㈣秴杞︺€侀噸鍨嬭溅杈嗙姝㈣秴杞﹀拰璺彛浼樺厛閫氳銆?
- 婕旂ず鏍锋湰缁忚繃妯″瀷閫愬紶楠岃瘉锛屽彂甯冩椂鍧囪兘寰楀埌姝ｇ‘ Top-1 缁撴灉銆?
- GUI 灏嗘紨绀哄尯鏍囬鏀逛负 `Class showcase`锛屾槑纭己璋冪被鍒鐩栥€?

## 1.0.0 - 2026-06-13

### Added

- 绾?C++ LeNet 椋庢牸 CNN锛歍ensor銆丆onv銆丷eLU銆丮axPooling銆丗latten銆丗C銆丼oftmax銆丩oss銆?
- mini-batch SGD 璁粌銆佹祴璇曢泦璇勪及銆佹ā鍨嬩繚瀛樹笌鍔犺浇銆?
- GTSRB 瀹樻柟鐩綍鍜?10 绫诲紑鍙戝瓙闆嗚鍙栥€?
- Qt Quick 妗岄潰鐣岄潰锛屾敮鎸佹墦寮€銆佹嫋鏀俱€侀瑙堝拰婕旂ず鍥剧墖銆?
- Top-1銆佺疆淇″害銆乀op-3 姒傜巼涓庢帹鐞嗚€楁椂鏄剧ず銆?
- 妯″瀷銆佸浘鐗囧拰缁撴灉涓夐樁娈电姸鎬佹彁绀恒€?
- 閿洏蹇嵎閿€佸伐鍏锋彁绀恒€佽繍琛屾椂淇℃伅鍜屽彲璇婚敊璇彁绀恒€?
- Windows CMake/Qt GitHub Actions 鎸佺画闆嗘垚銆?
- Windows x64 渚挎惡鍙戝竷鍖呫€佺増鏈枃浠跺拰 SHA-256 鏍￠獙鏂囦欢銆?

### Architecture

- 灏嗚繍琛屾椂璧勬簮鏌ユ壘闆嗕腑鍒?`ResourceLocator`銆?
- 灏?Top-K 鎺ㄧ悊鍜岃鏃堕泦涓埌 `InferenceEngine`銆?
- `AppController` 浠呭崗璋?GUI 鐘舵€併€佸紓姝ヤ换鍔″拰閿欒鍙嶉銆?

### Model

- 榛樿鍙戝竷妯″瀷涓?10 绫?GTSRB 瀛愰泦妯″瀷銆?
- 璁粌鏍锋湰 10,000锛屾祴璇曟牱鏈?5,670銆?
- 5 涓?Epoch 鍚庢祴璇曢泦 Top-1 鍑嗙‘鐜囦负 89.63%銆?

