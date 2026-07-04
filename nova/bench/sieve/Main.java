public class Main { public static void main(String[] a){ int n=2000000; byte[] s=new byte[n];
 for(int i=0;i<n;i++)s[i]=1; s[0]=0;s[1]=0;
 for(long i=2;i*i<n;i++) if(s[(int)i]==1) for(long j=i*i;j<n;j+=i) s[(int)j]=0;
 long c=0; for(int k=0;k<n;k++) c+=s[k]; System.out.println(c); } }
