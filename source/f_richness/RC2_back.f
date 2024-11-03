      parameter (ang=3.14159/180.)
      character fn1*20
      real az1(6),az2(9),z1(30000000,5)
      real sum(48),x(48,2),y(2000,3)
      integer(kind=8) id1,id2
      integer ip1(2),index1(30000000),ip2(2)
      open(30,file='../../output/f_richness/RC_back.dat')
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
      if(az1(6)+1.16*az1(4).le.-19.2.and.az1(5).le.0.08*(1+az1(4)))then
         N=N+1
         z1(N,1)=az1(1)
         z1(N,2)=az1(2)
         z1(N,3)=az1(4)
         z1(N,4)=az1(3)
         z1(N,5)=az1(6)-0.7745
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

      open(20,file='../../output/f_richness/RC_cluster_sz2nd1.dat')
 33   read(20,*,end=44)id2,az2,ip2
      if(az2(1).ge.alfmin.and.az2(1).le.alfmax.and.az2(2).ge.detmin
     $     .and.az2(2).le.detmax)then
c     -------------------------------
c     -------------------------------

      if(az2(4).ge.0.01)zz=az2(4)
      if(az2(4).lt.-0.01)zz=az2(3)
      if(zz.le.0.45)zgap=0.04*(1+zz)
      if(zz.gt.0.45)zgap=0.248*zz-0.0536
      do j=1,20
         angle=(j-1)*18.*ang
         x(j,1)=2.5*cos(angle)
         x(j,2)=2.5*sin(angle)
      enddo
      do j=21,48
         angle=(j-21)*12.8571*ang
         x(j,1)=3.5*cos(angle)
         x(j,2)=3.5*sin(angle)
      enddo
      ddz=dis(zz)
      kk=0
      do i=1,N
         if(index1(i).eq.1.and.abs(az2(1)-z1(i,1)).le.2.0.and.
     $        abs(az2(2)-z1(i,2)).le.2.0.and.ip2(2).ge.1.and.
     $        abs(zz-z1(i,3))/(1+zz).le.0.00833)then
            z7=abs(az2(1)-z1(i,1))**2*cos(az2(2)*ang)**2
     $           +abs(az2(2)-z1(i,2))**2
            z6=ddz*sqrt(z7)
            absmag=z1(i,5)+1.16*zz
c     ---------------------------------------------------------
            if(z6.ge.2..and.z6.le.4..and.absmag.le.-20.5.and.
     $           z1(i,4).gt.az2(9))then
               kk=kk+1
               tmp=0.4*(4.68-absmag)-10.
               y(kk,1)=(z1(i,1)-az2(1))*cos(ang*az2(2))*ddz
               y(kk,2)=(z1(i,2)-az2(2))*ddz
               y(kk,3)=0.5*10**tmp
            endif 
         endif

         if(index1(i).eq.1.and.abs(az2(1)-z1(i,1)).le.2.0.and.
     $        abs(az2(2)-z1(i,2)).le.2.0.and.ip2(2).eq.0.and.
     $        abs(zz-z1(i,3)).le.0.05)then
            z7=abs(az2(1)-z1(i,1))**2*cos(az2(2)*ang)**2
     $           +abs(az2(2)-z1(i,2))**2
            z6=ddz*sqrt(z7)
            absmag=z1(i,5)+1.16*zz
c     ---------------------------------------------------------
            if(z6.ge.2..and.z6.le.4..and.absmag.le.-20.5.and.
     $           z1(i,4).gt.az2(9))then
               kk=kk+1
               tmp=0.4*(4.68-absmag)-10.
               y(kk,1)=(z1(i,1)-az2(1))*cos(ang*az2(2))*ddz
               y(kk,2)=(z1(i,2)-az2(2))*ddz
               y(kk,3)=0.5*10**tmp
            endif 
         endif
         
         if(index1(i).eq.0.and.abs(az2(1)-z1(i,1)).le.2.0.and.
     $        abs(az2(2)-z1(i,2)).le.2.0.and.
     $        abs(zz-z1(i,3)).le.zgap)then
            z7=abs(az2(1)-z1(i,1))**2*cos(az2(2)*ang)**2
     $           +abs(az2(2)-z1(i,2))**2
            z6=ddz*sqrt(z7)
            absmag=z1(i,5)-dmode(zz,z1(i,3))+1.16*zz
c     ---------------------------------------------------------
            if(z6.ge.2..and.z6.le.4..and.absmag.le.-20.5.and.
     $           z1(i,4).gt.az2(9))then
               kk=kk+1
               tmp=0.4*(4.68-absmag)-10.
               y(kk,1)=(z1(i,1)-az2(1))*cos(ang*az2(2))*ddz
               y(kk,2)=(z1(i,2)-az2(2))*ddz
               y(kk,3)=0.5*10**tmp
            endif 
         endif
      enddo
c     -----------------------------------------------------------------
      do j=1,48
         sum(j)=0.
         do k=1,kk
            z9=(y(k,1)-x(j,1))**2+(y(k,2)-x(j,2))**2
            z8=sqrt(z9)
            if(z8.le.0.5)then
               sum(j)=sum(j)+y(k,3)
            endif
         enddo
      enddo
c     ------------------------------
      ave1=0.
      sdev1=0.
      do j=1,48
         ave1=ave1+sum(j)
      enddo
      ave1=ave1/48.0
      do j=1,48
         sdev1=sdev1+(sum(j)-ave1)**2
      enddo
      sdev1=sqrt(sdev1/47.)
      ave2=0.
      k1=0
      do j=1,48
         if(sum(j).le.ave1+3.0*sdev1)then
            k1=k1+1
            ave2=ave2+sum(j)
         endif
      enddo
      ave2=ave2/k1
c     ------------------------------
      write(30,100)id2,(az2(j),j=1,2),zz,ave2*4,kk

c     ---------------------------
c     ---------------------------
      endif
      goto 33
 44   close(20)

c      goto 15
c 16   close(12)
 100  format(I18,2F11.5,F8.4,F7.2,I4)
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

      function dmode(z1,z2)
      real z1,z2,dmode
      N1=100
      dz1=z1/N1
      sum1=0.0
      do j=1,N1
         sum1=sum1+dz1/sqrt(0.7+0.3*(1+dz1*j)**3)
      enddo
      sum1=sum1*(1+z1)

      N2=100
      dz2=z2/N2
      sum2=0.0
      do j=1,N2
         sum2=sum2+dz2/sqrt(0.7+0.3*(1+dz2*j)**3)
      enddo
      sum2=sum2*(1+z2)
      dmode=5*(log10(sum1)-log10(sum2))
      end


