
t = data.theta_meas;
xy = data.xy_meas_clean;
x = squeeze(xy(1, :, :, :));
y = squeeze(xy(2, :, :, :));

selectfigure theta
clf
p = panel();
p.pack(6)
for i = 1:6; p(i).select(); plot(squeeze(t(:, i, :))'); end

selectfigure xy
clf
p = panel();
p.pack(7, 2)
q = 0.6;
for i = 1:6
	p(i, 1).select(); plot(squeeze(x(:, i, :))');
	axis([0 5000 -q q])
	p(i, 2).select(); plot(squeeze(y(:, i, :))');
	axis([0 5000 -q q])
end
p(7, 1).select()
plot(squeeze(t(2, :, :))')
p(7, 2).select()
plot(squeeze(t(3, :, :))')



