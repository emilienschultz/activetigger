import { FC } from 'react';
import { FaStopCircle } from 'react-icons/fa';
import { Tooltip } from 'react-tooltip';
import { useGetServer, useStopProcesses } from '../../core/api';
import { useAuth } from '../../core/auth';
import { useNotifications } from '../../core/notifications';
import { ProjectStateModel } from '../../types';
import { ModalErrors } from '../ModalError';
import { PossibleProjectActions } from './ProjectPageLayout';

import { useAppContext } from '../../core/context';

/* define a component for project action bar 
with the project & the current action*/
export const StatusNotch: FC<{
  projectState: ProjectStateModel | null;
  currentProjectAction?: PossibleProjectActions;
  currentMode?: string;
  currentScheme?: string;
  currentUser: string;
  developmentMode?: boolean;
}> = ({
  projectState,
  currentUser,
  //  developmentMode,
}) => {
  // function to clear history
  const {
    appContext: { currentProject },
  } = useAppContext();
  const { authenticatedUser } = useAuth();
  const { stopProcesses } = useStopProcesses();
  // const nbUsers = projectState ? projectState.users.length : 0;

  // display the number of current processes on the server
  const { queueState, gpu } = useGetServer(currentProject || null);

  // 2 types of menu
  const onlyAnnotator = authenticatedUser?.status === 'annotator';

  // test if computation is currently undergoing
  const currentComputation =
    projectState && projectState.languagemodels
      ? currentUser in projectState.languagemodels.training ||
        currentUser in projectState.quickmodel.training ||
        currentUser in projectState.projections.training ||
        currentUser in projectState.bertopic.training ||
        Object.values(projectState.features.training).length > 0
      : false;

  // display the number of current processes on the server
  const { disk } = useGetServer(projectState || null);

  // notify if disk is full
  const { notify } = useNotifications();
  if (disk ? Number(disk['proportion']) > 98 : false) {
    notify({
      message: 'Disk is almost full, please delete some files or alert the admin',
      type: 'warning',
    });
  }

  const errors = projectState?.errors.map((arr) => arr.join(' - ')) || [];

  return (
    <div id="status-notch">
      {!onlyAnnotator && (
        <div style={{ display: 'flex' }}>
          {/* Display size of project 2 version (computer and smartphones) */}
          <span className="d-none d-md-inline">
            Size of the project:{' '}
            {projectState?.memory ? `${projectState.memory.toFixed(1)} Mo` : ''}
          </span>
          <span className="d-md-none">
            Size: {projectState?.memory ? `${projectState.memory.toFixed(1)} Mo` : ''}
          </span>
          {/* Display number of process running 1 version (computer) */}
          <span className="d-none d-md-inline">
            Processes running: {Object.values(queueState || []).length}
          </span>
          {/* Display GPU memory 1 version (computer) */}
          <span className="d-none d-md-inline">
            GPU Memory:
            {gpu
              ? ` ${(gpu['total_memory'] - gpu['available_memory']).toFixed(1)} / ${gpu['total_memory']} Go`
              : ' no GPU available'}
          </span>

          {/* Display GPU memory 1 version (computer) */}
          {currentComputation && (
            <>
              <span id="computing-span" style={{ display: 'flex', alignItems: 'center' }}>
                <div className="spinner-border spinner-border-sm text-warning" role="status">
                  <span className="visually-hidden">Computing</span>
                </div>
                <div>Computing</div>
                <a
                  id="stop-button"
                  onClick={() => stopProcesses('all')}
                  style={{ paddingBottom: '1.5px' }}
                >
                  <FaStopCircle style={{ color: 'red' }} />
                </a>
              </span>
              <Tooltip anchorSelect="#stop-button" place="top">
                Stop the current process
              </Tooltip>
            </>
          )}
          {/* Error pop up */}
          {projectState?.errors && projectState?.errors.length > 0 && (
            <ModalErrors errors={errors} />
          )}
        </div>
      )}
    </div>
  );
};
