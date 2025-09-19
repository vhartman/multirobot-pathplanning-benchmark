cp -f wsg_50_mesh_collision.urdf z.urdf
../../bin/urdf2ors.py > z.1.g
sed 's/package:\/\/wsg_50_description\/meshes/wsg_meshes/g' z.1.g > z.2.g
sed 's/\.obj/\.off/g' z.2.g > z.3.g
sed 's/color=\[0 0 0 0\]/color=\[0 0 0 1\]/g' z.3.g > z.4.g
#echo "Delete shape collision" >>z.5.g
../../bin/kinEdit -file z.4.g -cleanOnly
mv z.g z.wsg.g

# meshtools link_7.obj -fuse 0.001 -save
# meshtools WSG50_110.off -view -scale .001 -save
