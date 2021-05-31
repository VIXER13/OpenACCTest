# OpenACCTest
Лабораторные работы по OpenACC

pgcc -acc -ta=multicore program_name
pgcc -acc -ta=nvidia program_name

Запуск
mpisubmit.pl -t proc_count program_name
mpisubmit.pl -g gpus_count program_name
