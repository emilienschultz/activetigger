import { FC } from 'react';
import { useStopProcesses } from '../core/api';
import { LossChart } from './vizualisation/lossChart';
import { StopProcessButton } from './StopProcessButton';

export interface DisplayTrainingProcessesProps {
  projectSlug: string | null;
  processes:
    | {
        [key: string]:
          | {
              [key: string]:
                | string
                | number
                | {
                    [key: string]: unknown;
                  }
                | null
                | undefined;
            }
          | undefined;
      }
    | undefined;
  processStatus?: string;
  displayStopButton?: boolean;
}

interface LossData {
  epoch: { [key: string]: number };
  val_loss: { [key: string]: number };
  val_eval_loss: { [key: string]: number };
}

export const DisplayTrainingProcesses: FC<DisplayTrainingProcessesProps> = ({
  processes,
  processStatus,
  displayStopButton = false,
}) => {
  const { stopProcesses } = useStopProcesses();

  const displayAdvancement = (val: number | string | null) => {
    if (!val) return 'process in the queue waiting to start';
    const v = Math.round(Number(val));
    if (v >= 100) return 'completed, please wait';
    return v + '%';
  };

  if (
    processStatus &&
    processes &&
    Object.values(processes).filter((p) => p && p.status === processStatus).length === 0
  ) {
    return <div className="overflow-x-auto"></div>;
  }

  return (
    <div className="overflow-x-auto">
      {Object.keys(processes || {}).length > 0 && displayStopButton && <StopProcessButton />}
      {Object.keys(processes || {}).length > 0 && (
        <div>
          Process running:
          <ul className="list-group">
            {Object.entries(
              processes as Record<string, Record<string, string | number | null>>,
            ).map(([user, v]) => (
              <li className="list-group-item" key={v.name}>
                <div className="horizontal wrap" style={{ justifyContent: 'space-between' }}>
                  <span style={{ marginRight: '10px' }}>From: {user};</span>
                  <span style={{ marginRight: '10px' }}>Name: {v.name};</span>
                  <span style={{ marginRight: '10px' }}>Status: {v.status};</span>
                  <span style={{ marginRight: '10px' }} className="fw-bold">
                    {displayAdvancement(v.progress)}
                  </span>
                </div>
                {v.status === 'training' && (
                  <LossChart
                    loss={v.loss as unknown as LossData}
                    xmax={(v.epochs as number) || undefined}
                  />
                )}
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
