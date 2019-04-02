% character 1: m=man, w=woman, b=boy, g=girl
% characters 2-3: talker number
% characters 4-5: vowel (ae=”had”, ah=”hod”, aw=”hawed”, eh=”head”,
% er=”heard”, ei=”haid”, ih=”hid”, iy=”heed”, oa=/o/ as in “boat”,
% oo=”hood”, uh=”hud”, uw=”who'd”)
vowel = str2mat('aeæ','ah','aw','eh','er','ei','ih','iy','oa','oo','uh','uw');
talker_group = str2mat('m','w','b','g');

a = load('vowdata.dat')

filenames=char(filenames); % convert cell array to character matrix
[nfiles,nchar]=size(filenames);

for ifile=1:nfiles
    vowel_code(ifile) = strmatch(filenames(ifile,4:5),vowel);
    talker_group_code(ifile) = strmatch(filenames(ifile,1),talker_group);
    talker_number(ifile) = str2num(filenames(ifile,2:3));
end