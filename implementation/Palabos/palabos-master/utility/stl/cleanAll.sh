for i in *.cpp
do
    ed Makefile 1>/dev/null 2>/dev/null <<!
    g/^[ ]*projectFiles.*cpp/s//projectFiles = $i/
    w
    q
!
    make clean
done
/bin/rm -f .sconsign.dblite
