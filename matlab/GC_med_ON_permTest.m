addpath C:\code\spm12\
addpath C:\code\wjn_toolbox

%%
clear all
close all, clc
dt = readtable('decodingtimes.csv');
color_OFF = [55 110 180]./255;
color_ON = [113 188 173]./255;
color_DOT = [255 244 0]./255;
ioff = ci('OFF',dt.Medication);
ion = ci('ON',dt.Medication);
subs_off = dt.Subject(ioff);
subs_on = dt.Subject(ion);
fsample = 200;

% exclude sub-002
subs_off(ci('EL002',dt.Subject)) = [];


baseline =[]; % Do not correct baseline
time_window = [-2 0]; % Focus on post-baseline timewindow
freq_window = [4 33]; % Focus on 4 - 33 Hz for optimal visibility
sk = [0 .3.*fsample]; % Do not smooth frequencies, smooth time with 300 ms kernel
ds = 25; % Downsample to 100 ms resolution (200/20)=10 Hz


% Get time and frequency vectors
T = readtable(['sub-' subs_off{1} '_med-OFF_stim-OFF_net_trgc.csv']);
foi = 1+wjn_sc(table2array(T(2:end,4)),freq_window(1)):wjn_sc(table2array(T(2:end,4)),freq_window(2));
toi = 4+wjn_sc(table2array(T(1,5:end)),time_window(1)):wjn_sc(table2array(T(1,5:end)),time_window(2));


% Extract smooth and downsample OFF data, read decoding times
off_mat=[];soff_mat=[];rsoff_mat=[];dt_off=[];dsoff_mat=[];
for a = 1:length(subs_off)
    T = readtable(['sub-' subs_off{a} '_med-OFF_stim-OFF_net_trgc.csv']);
    t = table2array(T(1,toi));
    f = table2array(T(foi,4));
    off_mat(a,:,:) = table2array(T(foi,toi));
    soff_mat(a,:,:) = smooth2a(squeeze(off_mat(a,:,:)),sk(1),sk(2));
    if ~isempty(baseline)
        soff_mat(a,:,:)= wjn_raw_baseline(soff_mat(a,:,:),f,wjn_sc(t,baseline(1)):wjn_sc(t,baseline(2)));
    end
    if ~isempty(ds)
        dsoff_mat(a,:,:)=downsample(squeeze(soff_mat(a,:,:))',ds)';
        t=downsample(t,ds);
    end
    dt_off(a,1) = dt.EarliestTimepoint(intersect(ci(subs_off{a},dt.Subject),ci('OFF',dt.Medication)));
end
 if ~isempty(ds)
    soff_mat=dsoff_mat;
 end

 % Extract smooth and downsample ON data, read decoding times
on_mat=[];son_mat=[];dt_on=[];dson_mat=[];
for a = 1:length(subs_on)
    T = readtable(['sub-' subs_on{a} '_med-ON_stim-OFF_net_trgc.csv']);
    t = table2array(T(1,toi));
    f = table2array(T(foi,4));
    on_mat(a,:,:) = table2array(T(foi,toi));
    son_mat(a,:,:) = smooth2a(squeeze(on_mat(a,:,:)),sk(1),sk(2));
    if ~isempty(baseline)
        son_mat(a,:,:)= wjn_raw_baseline(son_mat(a,:,:),f,wjn_sc(t,baseline(1)):wjn_sc(t,baseline(2)));
    end
    if ~isempty(ds)
        dson_mat(a,:,:)=downsample(squeeze(son_mat(a,:,:))',ds)';
        t=downsample(t,ds);
    end
    dt_on(a,1) = dt.EarliestTimepoint(intersect(ci(subs_on{a},dt.Subject),ci('ON',dt.Medication)));
end
if ~isempty(ds)
    son_mat=dson_mat;
end



% Test pixel-wise - consider right tailed only for cleaner results
p=[];
for a = 1:size(soff_mat,2)
    for b = 1:size(soff_mat,3)
        if nanmedian(son_mat(:,a,b))>nanmedian(soff_mat(:,a,b))
            p(a,b)=wjn_pt(son_mat(:,a,b),soff_mat(:,a,b))/2;
        else
            p(a,b)=1;
        end
%         p(a,b)=ranksum(son_mat(:,a,b),soff_mat(:,a,b));
    end
end
op=p;

original_m=[];m=[];
for a = 1:10000
    if a==1
        cc_p=bwconncomp(p<.05);
    else
        p(:) = p(randperm(length(p(:))));
        cc_p=bwconncomp(p<.05);
    end
        idx = cc_p.PixelIdxList;
    rsum = [];
        for b = 1:length(idx)
            rsum(b) = sum(1-p(idx{b}));
        end
        [m,i]=nanmax(rsum);

        if a == 1
            original_m = m;
            sigpixel=idx(i);
        else
            surrogate_m(a-1) = m;
        end
end


pcluster = 1-(wjn_sc(sort(surrogate_m),original_m)./10000)
if pcluster <=0.05
pp = ones(size(p));
pp(sigpixel{1})=pcluster;
end
p=op;
figure,
wjn_contourf(t,f,nanmean(son_mat)-nanmean(soff_mat),200)
hold on
wjn_contourp(t,f,squeeze(pp)<=0.05)
caxis([-0.02 0.04]);
ylabel('Frequency [Hz]')
xlabel('Time [s]')
colormap('viridis')
figone(7)
%
title('ON-OFF')
myprint('20231031_GC_off-on_no_sub002')
colorbar
myprint('20231031_GC_off-on_no_sub002_cb')


figure,
[x,y]=find(p==min(p,[],'all'));
wjn_contourf(t,f,-(pp<0.05),1)
hold on
plot([nanmean(dt_off) nanmean(dt_off)],[freq_window(1) freq_window(2)],'linestyle','--','color',color_OFF,'linewidth',2)
plot([nanmean(dt_on) nanmean(dt_on)],[freq_window(1) freq_window(2)],'linestyle','--','color',color_ON, 'linewidth',2)
scatter(t(y(1)),f(x(1)),'filled','square','markerfacecolor',color_DOT)
colormap('gray')
hold on
ylabel('Frequency [Hz]')
xlabel('Time [s]')
figone(7)
% title('ON-OFF')
myprint('20231031_GC_p_off-on_no_sub002')
colorbar
myprint('20231031_GC_p_off-on_no_sub002_cb')

% Correlate decoding timepoint at peak significance OFF-ON

figure
wjn_corr_plot(squeeze(soff_mat(:,x,y)),dt_off,color_OFF,0)
% title('OFF GC ~ OFF Decoding Time')
ylabel('Decoding time [s]')
xlabel('Granger causality')
set(gcf,'position',[1480         874         355         364])
myprint('20231031_GC_DT_OFF_corr_no_sub002')

figure
wjn_corr_plot(squeeze(son_mat(:,x,y)),dt_on,color_ON,0)
% title('ON GC ~ ON Decoding Time')
ylabel('Decoding time [s]')
xlabel('Granger causality')
set(gcf,'position',[1480         874         355         364])
xlim([-0.06 0.2])
ylim([-1.75 -0.35])
myprint('20231031_GC_DT_ON_corr_no_sub002')


%% Plot granger spectra at -2 s and peak significance

psoff=[];
for a = 1:size(soff_mat,2)
    psoff(a)=wjn_pt(squeeze(nanmean(soff_mat(:,a,wjn_sc(t,-2)),3))',squeeze(nanmean(son_mat(:,a,wjn_sc(t,-2)),3))')
end

figure
mypower(f,squeeze(nanmean(soff_mat(:,:,wjn_sc(t,-2)),3))',color_OFF);
hold on
mypower(f,squeeze(nanmean(son_mat(:,:,wjn_sc(t,-2)),3))',color_ON);
ylim([-0.02 0.08])
figone(7)
xlabel('Frequency [Hz]')
ylabel('Granger causality')
sigbar(f,psoff<0.05)
mypower(f,squeeze(nanmean(soff_mat(:,:,wjn_sc(t,-2)),3))',color_OFF);
hold on
mypower(f,squeeze(nanmean(son_mat(:,:,wjn_sc(t,-2)),3))',color_ON);
legend('OFF','ON')
title({'Granger causality at end of baseline';'(-2 s)'})
myprint('20231031_GC_rest_OFF_ON_no_sub002')

pson=[];
for a = 1:size(son_mat,2)
    pson(a)=wjn_pt(squeeze(nanmean(soff_mat(:,a,y(1)),3))',squeeze(nanmean(son_mat(:,a,y(1)),3))');
end

figure
mypower(f,squeeze(soff_mat(:,:,y(1)))',color_OFF);
hold on
mypower(f,squeeze(son_mat(:,:,y(1)))',color_ON);
ylim([-0.02 0.08])
figone(7)
xlabel('Frequency [Hz]')
ylabel('Granger causality')
sigbar(f,pson<0.05)
mypower(f,squeeze(soff_mat(:,:,y(1)))',color_OFF);
hold on
mypower(f,squeeze(son_mat(:,:,y(1)))',color_ON);
legend('OFF','ON')
title({'Granger causality at peak difference';'(-0.9 s)'})
myprint('20231031_GC-peak_OFF_ON_no_sub002')
