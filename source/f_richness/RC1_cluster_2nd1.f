      parameter (ang=3.14159265/180.)
      character fn1*20
c     az2(n+1): old version file used to compare
      real az1(6),az2(6),zmag(300)
      real z1(30000000,4)
      integer(kind=8) id1,id2
      integer ip1(2),index1(30000000),ip2(2)
      open(30,file='../../output/f_richness/RC_cluster_sz2nd1.dat')
c      open(12,file='../cand_bcg2nd.lst')
c 15   read(12,*,end=16)fn1
c      open(10,file='../'//fn1)
      open(10,file='../f_gsm/plot_data/merge_gcap.dat')
      alfmin=360.
      alfmax=0.
      detmin=90.
      detmax=-90.
      N=0
 11   read(10,*,end=22)id1,az1,ip1
      if(az1(6)+1.16*az1(4).le.-19.73.and.az1(5).le.0.08*(1+az1(4)))then
         N=N+1
c        ra 
         z1(N,1)=az1(1)
c        dec
         z1(N,2)=az1(2)
c        z
         z1(N,3)=az1(4)
c        mag
         z1(N,4)=az1(3)
c        is_spec 
         index1(N)=ip1(2)
      endif

      if(az1(1).lt.alfmin)then
         alfmin=az1(1)
      endif
      if(az1(1).gt.alfmax)then
         alfmax=az1(1)
      endif
      if(az1(2).lt.detmin)then
         detmin=az1(2)
      endif
      if(az1(2).gt.detmax)then
         detmax=az1(2)
      endif
      goto 11
 22   close(10)

      write(*,'(i7,4f6.1)')N,alfmin,alfmax,detmin,detmax

      open(20,file='../../output/f_richness/predcen4calc.dat')
 33   read(20,*,end=44)id2,az2,ip2
      if(az2(1).ge.alfmin.and.az2(1).le.alfmax.and.az2(2).ge.detmin
     $     .and.az2(2).le.detmax)then
c     -------------------------------
c     -------------------------------
      if(az2(4).ge.0.01)zz=az2(4)
      if(az2(4).lt.-0.01)zz=az2(3)
      if(zz.le.0.45)zgap=0.04*(1+zz)
      if(zz.gt.0.45)zgap=0.248*zz-0.0536
      k2=0
      do i=1,N
         if(index1(i).eq.1.and.abs(az2(1)-z1(i,1)).le.0.5.and.
     $        abs(az2(2)-z1(i,2)).le.0.5.and.ip2(2).ge.1.and.
     $        abs(zz-z1(i,3))/(1+zz).le.0.00833)then
            z7=abs(az2(1)-z1(i,1))**2*cos(az2(2)*ang)**2
     $           +abs(az2(2)-z1(i,2))**2
            z6=dis(zz)*sqrt(z7)
            if(z6.le.1.0)then
               k2=k2+1
               zmag(k2)=z1(i,4)
            endif
         endif

         if(index1(i).eq.1.and.abs(az2(1)-z1(i,1)).le.0.5.and.
     $        abs(az2(2)-z1(i,2)).le.0.5.and.ip2(2).eq.0.and.
     $        abs(zz-z1(i,3)).le.0.05)then
            z7=abs(az2(1)-z1(i,1))**2*cos(az2(2)*ang)**2
     $           +abs(az2(2)-z1(i,2))**2
            z6=dis(zz)*sqrt(z7)
            if(z6.le.1.0)then
               k2=k2+1
               zmag(k2)=z1(i,4)
            endif
         endif

         if(index1(i).eq.0.and.abs(az2(1)-z1(i,1)).le.0.5.and.
     $        abs(az2(2)-z1(i,2)).le.0.5.and.
     $        abs(zz-z1(i,3)).le.zgap)then
            z7=abs(az2(1)-z1(i,1))**2*cos(az2(2)*ang)**2
     $           +abs(az2(2)-z1(i,2))**2
            z6=dis(zz)*sqrt(z7)
            if(z6.le.1.0)then
               k2=k2+1
               zmag(k2)=z1(i,4)
            endif
         endif
      enddo
      chi2=99.9
      do j=1,k2
         if(zmag(j).gt.az2(4)+0.02.and.zmag(j).le.chi2)then
            chi2=zmag(j)
         endif
      enddo
      write(30,100)id2,(az2(j),j=1,5),-1.0,-1.0,az2(6),chi2,ip2

c     ---------------------------
c     ---------------------------
      endif
      goto 33
 44   close(20)

c      goto 15
c 16   close(12)
 100  format(I18,2F11.5,2F8.4,F7.2,F6.2,2F7.2,F7.2,2I4)
      end

      function dis(z)
      real dis,z
      N_step=100
      dz=z/N_step
      sum=0.0
      do j=1,N_step
         sum=sum+dz/sqrt(0.7+0.3*(1+dz*j)**3)
      enddo
      dis=4285.7*sum/(1+z)/57.3
      end
