c     The programe use a NFW profile to smooth member galaxy distribution,
c     similar with "clust_mapg.f". The difference is the code use a radius
c     depending smooth scale.
c     --------------------------parameters---------------------------
      parameter (ang=3.14159265/180.)
      character cid*6
      character fid*2
      real az1(6),az2(7)
      real z3(20000000),z4(20000000),z5(20000000)
      real zmagabs(20000000), is_spec(20000000)
      real y(5000,3),b(200,200)
      real ra_u,ra_l,dec_u,dec_l
      integer id1,id2,ip1(2),fnum
      integer filelimit
      real fdata(5),ra,dec,tmp,lum
c     ----------------------open and load files---------------------
      fnum = 0
      filelimit=71

 10   fnum = fnum + 1

      write(fid,'(i2.2)')fnum
      open(10,file='plot_data/merge_gcap.dat')
      open(20,file='../../output/fgsm_input.dat')
      open(30,file='plot_data/readme_table.txt')
      N=0
 11   read(10,*,end=22)id1,az1,ip1
      N=N+1
      z3(N)=az1(1)
      z4(N)=az1(2)
      z5(N)=az1(4)    
      zmagabs(N)=-az1(6) 
      is_spec(N)=ip1(2)
      goto 11
 22   close(10)

 23   read(30,*,end=24)fdata
      if(fdata(1).eq.fnum)then
        ra_l = fdata(2)
        ra_u = fdata(3)
        dec_l = fdata(4)
        dec_u = fdata(5)
      endif
      goto 23
 24   close(30)

      kk=0
 33   read(20,*,end=44)id2,az2
c     ----------------choose brightest galaxies----------------------
      ra = az2(1)
      dec = az2(2) 
      z = az2(3)
      r_som = az2(4)
      sig_som = az2(5)
      photz_slice = az2(6)
      specz_slice = az2(7)
      if(ra.ge.ra_l.and.ra.le.ra_u)then
      if(dec.ge.dec_l.and.dec.le.dec_u)then

      ddz=dis(z)
      M=0
      do i=1,N
         if (is_spec(i) == 0)then
            delta_z = photz_slice
         else
            delta_z = specz_slice
         endif
         
         if(abs(ra-z3(i)).le.0.5.and.abs(dec-z4(i)).le.0.5
     $        .and.abs(z-z5(i)).le.delta_z)then

           z7=abs(ra-z3(i))**2*cos(dec*ang)**2+
     $            abs(dec-z4(i))**2
           z6=ddz*sqrt(z7)

            if(z6.le.r_som)then
               M=M+1
               y(M,1)=-1.0*(z3(i)-ra)*cos(ang*dec)*ddz
               y(M,2)=-1.0*(z4(i)-dec)*ddz
               am = -zmagabs(i)+1.16*z
               tmp=0.4*(4.68-am)-10.
               lum=0.5*10**tmp
               y(M,3)=lum
            endif
         endif
      enddo
c     --------------------------gauss smooth-----------------------
      
      do i=1,200
         xi= -1*r_som + i*0.01*r_som
         do j=1,200
            yj= -1*r_som + j*0.01*r_som
            sum=0.
            do k=1,M
               dd=sqrt((y(k,1)-xi)**2+(y(k,2)-yj)**2)
               dx=sqrt(xi**2+yj**2)
               if(dd.le.r_som)then
                  sum=sum+y(k,3)*gauss(dd, sig_som)
               endif
            enddo
            b(i,j)=sum
         enddo
      enddo
c     ----------------------positive pictures----------------------
      write(cid,'(i6.6)')id2
      open(1,file='../../output/som/'//cid//'.dat')
      do i = 1,200
        write(1,*)b(:,i)
      enddo
      close(1)
      write(40,*)id2
      
      kk=kk+1
      endif
      endif
      goto 33
 44   close(20)

      if(fnum.lt.filelimit)then
        goto 10
      endif
       
      end

c     ------------------functions-------------------------
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

      function gauss(x, sig)
      real gauss,x,sig
      gauss=exp(-0.5*x**2/sig**2)/(2*3.14159*sig**2)
      end
