import { FC } from 'react';
import PulseLoader from 'react-spinners/PulseLoader';
import { useStopProcesses } from '../core/api';

interface EmptyProps {}

export const StopProcessButton: FC<EmptyProps> = ({}) => {
  const { stopProcesses } = useStopProcesses();
  return (
    <button key="stop" className="btn-stop-process" onClick={() => stopProcesses('bert')}>
      <PulseLoader color={'white'} /> Stop process
    </button>
  );
};
