#ifndef IMPLICIT_EULER_HPP
#define IMPLICIT_EULER_HPP

#include "../machineEpsilon/maceps.hpp"
#include "../newtonsMethod/newtonsMethod.hpp"
#include <iomanip>
#include <iostream>

template <typename T, typename F>
implicit_euler (F f,T df,T x0,T t0,T tf,const unsigned int MAX_ITERATIONS)
{
  auto h=(tf-t0)/n;
  auto t=linspace(t0,tf,n+1);
  auto y=zeros(n+1,length(y0));
  auto y(1,:)=y0;
  for (auto i=0u; i < MAX_ITERATIONS; ++i)
  {
    x0=y(i,:)’;
    x1=x0-inv(eye(length(y0))-h*feval(df,t(i),x0))*(x0-h*feval(f,t(i),x0)’-y(i,:)’);
    x1 = newtons_method(f, dt, x0)
    y(i+1,:)=x1’;
  }
}

#endif
