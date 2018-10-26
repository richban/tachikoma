--inicialize
dc=0 --distance cumulative
pp = {0, 0, 0}

threadFunction=function()
   while true do
		--get actual position
		bodyHandle=simGetObjectHandle('Pioneer_p3dx#0')
        
        if bodyHandle == -1 then simAddStatusbarMessage('Body not found') end
  		
        p=simGetObjectPosition(bodyHandle,-1)

 		--distance travelled from previous position to actual position and increment distance cumulative
		d=math.sqrt(((p[1]-pp[1])^2)+((p[2]-pp[2])^2))
 		dc=dc+d

		--update position 
		pp[1]=p[1]
		pp[2]=p[2]
 		pp[3]=p[3]
        
 		--send signal to matlab
 		simSetFloatSignal('distance', dc)

      simSwitchThread()
      -- The thread resumes in next simulation step
   end
end

simSetThreadSwitchTiming(2000)

res,err=xpcall(threadFunction,function(err) return debug.traceback(err) end)
if not res then
   simAddStatusbarMessage('Lua runtime error: '..err)
end
