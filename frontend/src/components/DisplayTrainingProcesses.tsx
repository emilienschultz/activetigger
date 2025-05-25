import { FC } from 'react';
import PulseLoader from 'react-spinners/PulseLoader';
import { useStopTrainBertModel } from '../core/api';
import { LossChart } from './vizualisation/lossChart';

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
}

interface LossData {
  epoch: { [key: string]: number };
  val_loss: { [key: string]: number };
  val_eval_loss: { [key: string]: number };
}

export const DisplayTrainingProcesses: FC<DisplayTrainingProcessesProps> = ({
  projectSlug,
  processes,
}) => {
  const { stopTraining } = useStopTrainBertModel(projectSlug || null);

  const displayAdvancement = (val: number | string | null) => {
    if (!val) return 'process in the queue waiting to start';
    const v = Math.round(Number(val));
    if (v >= 100) return 'completed, please wait';
    return v + '%';
  };

  return (
    <div className="overflow-x-auto p-4">
      {Object.keys(processes || {}).length > 0 && (
        <div>
          <button
            key="stop"
            className="btn btn-primary mt-3 d-flex align-items-center"
            onClick={stopTraining}
          >
            <PulseLoader color={'white'} /> Stop current process
          </button>
        </div>
      )}
      {Object.keys(processes || {}).length > 0 && (
        <div className="mt-3">
          Current process:
          <ul>
            {Object.entries(
              processes as Record<string, Record<string, string | number | null>>,
            ).map(([_, v]) => (
              <li key={v.name}>
                {v.name} - {v.status} :{' '}
                <span style={{ fontWeight: 'bold' }}>
                  {displayAdvancement(v.progress)}
                  {v.status === 'training' && (
                    <LossChart
                      loss={v.loss as unknown as LossData}
                      xmax={(v.epochs as number) || undefined}
                    />
                  )}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
