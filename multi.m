function [outputArg1] = multi(T,ita,nbi)
%MULTI Summary of this function goes here
A=exmeet(ita,nbi,1);
                for i=2:T-1
                    A=A+exmeet(ita,nbi,i);
                end
outputArg1 = A;

end

 function f=exmeet(ita,nbi,t)
        f=(ita*nbi)^t;
 end