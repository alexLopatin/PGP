
$step = $args[1] -as [int]
$max = $args[2] -as [int]

$projectName = $args[0]

#генерация тестов
For ($i=$step; $i -le $max; $i+= $step) {
    py test.py $i
}

if(Test-Path "tests/CPU.txt")
{
    Remove-Item -Path "tests/CPU.txt"
}
#запуск тестов
For ($i=$step; $i -le $max; $i+= $step) {
    $test = Get-Content "./tests/test$($i).txt"
    $result = $test | & "./../x64/CPU/$($projectName).exe"
    Add-Content -Path "tests/CPU.txt" -Value ($result -split '\n')[0]
}

if(Test-Path "tests/GPU.txt")
{
    Remove-Item -Path "tests/GPU.txt"
}
#запуск тестов
For ($i=$step; $i -le $max; $i+= $step) {
    $test = Get-Content "./tests/test$($i).txt"
    $result = $test | & "./../x64/GPU/$($projectName).exe"
    Add-Content -Path "tests/GPU.txt" -Value ($result -split '\n')[0]
}