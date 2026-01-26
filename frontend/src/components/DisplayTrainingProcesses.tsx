import { FC } from 'react';
// import { useStopProcesses } from '../core/api';
import { StopProcessButton } from './StopProcessButton';
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
  const formatProgress = (val: number | string | null) => {
    if (val === null || val === undefined) {
      return { label: 'Waiting in queue', value: 0 };
    }
    const v = Math.round(Number(val));
    if (v >= 100) {
      return { label: 'Finalizing…', value: 100 };
    }
    return { label: `${v}%`, value: v };
  };

  if (
    processStatus &&
    processes &&
    Object.values(processes).filter((p) => p && p.status === processStatus).length === 0
  ) {
    return <div className="overflow-x-auto" />;
  }

  const entries = Object.entries(
    processes as Record<string, Record<string, string | number | null>>,
  );

  if (entries.length === 0) return null;

  return (
    <div className="overflow-x-auto my-4">
      {displayStopButton && (
        <div className="mb-3">
          <StopProcessButton projectSlug={projectSlug} />
        </div>
      )}

      <h5 className="mb-3">Running processes</h5>

      <div className="list-group">
        {entries.map(([user, v]) => {
          const progress = formatProgress(v.progress);

          return (
            <div key={v.name as string} className="list-group-item">
              {/* Header */}
              <div className="d-flex justify-content-between align-items-start mb-2">
                <div>
                  <div className="fw-bold">{v.name}</div>
                  <div className="text-muted small">
                    From {user} · Status: {v.status}
                  </div>
                </div>

                <span className="fw-semibold">{progress.label}</span>
              </div>

              {/* Progress bar */}
              <div className="progress mb-2" style={{ height: 6 }}>
                <div
                  className="progress-bar"
                  role="progressbar"
                  style={{ width: `${progress.value}%` }}
                  aria-valuenow={progress.value}
                  aria-valuemin={0}
                  aria-valuemax={100}
                />
              </div>

              {/* Training details */}
              {v.status === 'training' && (
                <div className="mt-3">
                  <LossChart
                    loss={v.loss as unknown as LossData}
                    xmax={(v.epochs as number) || undefined}
                  />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
