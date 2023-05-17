# shell script to source things and define variables
# almost identical to backup.sh, but doesn't start server

source ~/root/bin/thisroot.sh
source csctimingenv/bin/activate
export CACHE=/root/.globus/
export CACERT=/root/.globus/CERN_Cert_2.crt
export PUBLIC_KEY=/root/.globus/usercert.pem
export PRIVATE_KEY=/root/.globus/userkey.pem
update-ca-trust
