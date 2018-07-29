# sync with remote server, execute on server and then pull result back.
rm model.h5
rsync -a /Users/attwell/Uber/GitHub/CarND-Behavioral-Cloning-P3 attwell@172.30.234.67:/home/attwell/GitHub/
/usr/bin/ssh -i /Users/attwell/.ssh/id_test attwell@172.30.234.67 'bash -s' < execute_model.sh
scp -i /Users/attwell/.ssh/id_test attwell@172.30.234.67:/home/attwell/GitHub/CarND-Behavioral-Cloning-P3/model.h5 .