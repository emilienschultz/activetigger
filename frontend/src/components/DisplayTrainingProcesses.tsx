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
  processStatus?: string;
  displayStopButton?: boolean;
}

interface LossData {
  epoch: { [key: string]: number };
  val_loss: { [key: string]: number };
  val_eval_loss: { [key: string]: number };
}

export const DisplayTrainingProcesses: FC<DisplayTrainingProcessesProps> = ({
  projectSlug,
  processes,
  processStatus,
  displayStopButton = false,
}) => {
  const { stopTraining } = useStopTrainBertModel(projectSlug || null);

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
      {Object.keys(processes || {}).length > 0 && displayStopButton && (
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
          Process running:
          <ul className="list-group">
            {Object.entries(
              processes as Record<string, Record<string, string | number | null>>,
            ).map(([user, v]) => (
              <li className="list-group-item" key={v.name}>
                <div className="d-flex justify-content-between align-items-center">
                  <div>
                    <strong>From:</strong> {user} <br />
                    <strong>Name:</strong> {v.name} <br />
                    <strong>Status:</strong> {v.status}
                  </div>
                  <div className="text-end">
                    <span className="fw-bold">{displayAdvancement(v.progress)}</span>
                    {v.status === 'training' && (
                      <div className="mt-2">
                        <LossChart
                          loss={v.loss as unknown as LossData}
                          xmax={(v.epochs as number) || undefined}
                        />
                      </div>
                    )}
                  </div>
                </div>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};
