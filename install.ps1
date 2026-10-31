# ติดตั้งทุกอย่างในโฟลเดอร์นี้ รันซ้ำได้ ขั้นที่เสร็จแล้วจะถูกข้าม เรียกผ่าน install.bat
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$TrackNet = Join-Path $Root 'third_party\TrackNetV3'
$TrackNetCommit = '6eda442ada1740573f200f836d93edc9a541ee86'
$CkptsDriveId = '1CfzE87a0f6LhBp0kniSl1-89zaLCZ8cA'

function Step($text) { Write-Host "`n=== $text ===" -ForegroundColor Cyan }
function Fail($text) { Write-Host "`n[ติดตั้งไม่สำเร็จ] $text" -ForegroundColor Red; exit 1 }
function Run($what, $exe) {
    # exe กับ args ส่งมาเป็น array; เช็ก exit code เองเพราะ native command ไม่โยน error
    & $exe[0] $exe[1..($exe.Count - 1)]
    if ($LASTEXITCODE -ne 0) { Fail "$what (exit code $LASTEXITCODE)" }
}
function Has($name) { [bool](Get-Command $name -ErrorAction SilentlyContinue) }

Step 'ตรวจเครื่อง'
if (-not (Has 'git')) {
    Fail "ไม่พบ Git ติดตั้งด้วย: winget install Git.Git แล้วเปิดหน้าต่างใหม่และรันอีกครั้ง"
}
$py = $null
if (Has 'py') {
    try {
        & py -3.11 --version 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) { $py = @('py', '-3.11') }
    } catch {}
}
if (-not $py -and (Has 'python')) {
    try {
        $v = & python -c "import sys;print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($v -eq '3.11') { $py = @('python') }
    } catch {}
}
if (-not $py) {
    Fail "ไม่พบ Python 3.11 ติดตั้งด้วย: winget install Python.Python.3.11 แล้วเปิดหน้าต่างใหม่และรันอีกครั้ง"
}
$hasNvidia = Has 'nvidia-smi'
if ($hasNvidia) {
    $torchIndex = 'https://download.pytorch.org/whl/cu121'
    Write-Host 'พบการ์ดจอ NVIDIA: ติดตั้ง PyTorch รุ่น GPU'
} else {
    $torchIndex = 'https://download.pytorch.org/whl/cpu'
    Write-Host 'ไม่พบการ์ดจอ NVIDIA: ติดตั้ง PyTorch รุ่น CPU ใช้งานได้แต่จะช้ามาก' -ForegroundColor Yellow
}

Step 'โหลดโค้ด TrackNetV3'
if (Test-Path (Join-Path $TrackNet 'predict.py')) {
    Write-Host 'มีอยู่แล้ว ข้าม'
} elseif (Test-Path (Join-Path $Root '.git')) {
    Run 'โหลด submodule' @('git', '-C', $Root, 'submodule', 'update', '--init')
} else {
    # โหลดโปรเจกต์มาเป็น ZIP จึงไม่มี submodule
    if (Test-Path $TrackNet) { Remove-Item $TrackNet -Recurse -Force }
    Run 'clone TrackNetV3' @('git', 'clone', 'https://github.com/qaz812345/TrackNetV3.git', $TrackNet)
    Run 'เลือก commit ของ TrackNetV3' @('git', '-C', $TrackNet, 'checkout', '-q', $TrackNetCommit)
}

Step 'ติดตั้ง library ของโปรเจกต์ (ครั้งแรกใช้เวลานาน)'
$venv = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $venv)) { Run 'สร้าง .venv' ($py + @('-m', 'venv', (Join-Path $Root '.venv'))) }
# ต้องลง torchvision จาก index เดียวกัน ไม่อย่างนั้น ultralytics จะดึงรุ่น CPU จาก PyPI มาทับ
Run 'ติดตั้ง PyTorch' @($venv, '-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', $torchIndex)
Run 'ติดตั้ง requirements' @($venv, '-m', 'pip', 'install', '-r', (Join-Path $Root 'requirements.txt'))

Step 'ติดตั้ง library ของ TrackNetV3 (ครั้งแรกใช้เวลานาน)'
$tnPy = Join-Path $TrackNet 'env\Scripts\python.exe'
if (-not (Test-Path $tnPy)) { Run 'สร้าง env ของ TrackNetV3' ($py + @('-m', 'venv', (Join-Path $TrackNet 'env'))) }
Run 'ติดตั้ง PyTorch' @($tnPy, '-m', 'pip', 'install', 'torch', '--index-url', $torchIndex)
Run 'ติดตั้ง library' @($tnPy, '-m', 'pip', 'install', 'dash', 'numpy', 'opencv_python', 'pandas', 'Pillow', 'plotly', 'parse', 'tqdm', 'pycocotools', 'gdown')

Step 'โหลดไฟล์โมเดล TrackNetV3'
$ckpts = Join-Path $TrackNet 'ckpts'
$need = @('TrackNet_best.pt', 'InpaintNet_best.pt')
$missing = $need | Where-Object { -not (Test-Path (Join-Path $ckpts $_)) }
if (-not $missing) {
    Write-Host 'มีอยู่แล้ว ข้าม'
} else {
    $tmp = Join-Path $TrackNet '_ckpts_tmp'
    if (Test-Path $tmp) { Remove-Item $tmp -Recurse -Force }
    New-Item -ItemType Directory -Force $tmp, $ckpts | Out-Null
    $zip = Join-Path $tmp 'ckpts.zip'
    & $tnPy -m gdown $CkptsDriveId -O $zip
    if ($LASTEXITCODE -ne 0) {
        Fail "โหลดจาก Google Drive ไม่ได้ ให้โหลดเองจาก https://drive.google.com/file/d/$CkptsDriveId/view แล้วแตกไฟล์ .pt ไว้ที่ $ckpts จากนั้นรันอีกครั้ง"
    }
    Expand-Archive $zip -DestinationPath $tmp
    foreach ($n in $need) {
        $f = Get-ChildItem $tmp -Recurse -Filter $n | Select-Object -First 1
        if (-not $f) { Fail "ไม่พบ $n ในไฟล์ที่โหลดมา" }
        Move-Item $f.FullName (Join-Path $ckpts $n) -Force
    }
    Remove-Item $tmp -Recurse -Force
}

Step 'ตรวจผลการติดตั้ง'
Run 'เรียก library ของโปรเจกต์' @($venv, '-c', "import torch, cv2, pandas, pyarrow, PIL, ultralytics; print('project ok, GPU =', torch.cuda.is_available())")
Run 'เรียก library ของ TrackNetV3' @($tnPy, '-c', "import torch, cv2, pandas, parse, tqdm; print('tracknet ok, GPU =', torch.cuda.is_available())")
if ($hasNvidia) {
    $gpu = & $venv -c "import torch;print(torch.cuda.is_available())"
    if ($gpu -ne 'True') { Write-Host 'เตือน: มีการ์ด NVIDIA แต่ PyTorch ใช้ GPU ไม่ได้ ลองอัปเดตไดรเวอร์การ์ดจอ' -ForegroundColor Yellow }
}

Write-Host "`nติดตั้งเสร็จแล้ว" -ForegroundColor Green
if (-not (Has 'ffmpeg')) {
    Write-Host 'ยังขาด ffmpeg ติดตั้งด้วย: winget install Gyan.FFmpeg แล้วเปิดหน้าต่างใหม่' -ForegroundColor Yellow
}
Write-Host 'วิธีใช้งานดูหัวข้อ 3 และ 4 ใน README.md'
