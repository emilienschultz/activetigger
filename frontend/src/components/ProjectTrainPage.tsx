import { FC, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { MdOutlineDeleteOutline } from 'react-icons/md';
import { useNavigate, useParams } from 'react-router-dom';

import { useDeleteBertModel, useRenameBertModel, useTrainBertModel } from '../core/api';
import { useAuth } from '../core/auth';
import { useAppContext } from '../core/context';
import { useNotifications } from '../core/notifications';
import { newBertModel } from '../types';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

/**
 * Component to manage model training
 */

interface renameModel {
  new_name: string;
}

export const ProjectTrainPage: FC = () => {
  const { projectName: projectSlug } = useParams();
  const { authenticatedUser } = useAuth();
  const navigate = useNavigate();

  const { notify } = useNotifications();
  const {
    appContext: { currentScheme, currentProject: project },
    setAppContext,
  } = useAppContext();

  if (!projectSlug) return null;
  if (!currentScheme) {
    notify({ type: 'warning', message: 'You need to select first a scheme' });
    navigate(`/projects/${projectSlug}`);
    return null;
  }

  const [currentModel, setCurrentModel] = useState<string | null>(null);

  // available models
  const availableModels = project?.bertmodels.available[currentScheme]
    ? Object.keys(project?.bertmodels.available[currentScheme])
    : [];

  const { deleteBertModel } = useDeleteBertModel(projectSlug);

  // form to rename
  const { renameBertModel } = useRenameBertModel(projectSlug);
  const {
    handleSubmit: handleSubmitRename,
    register: registerRename,
    reset: resetRename,
  } = useForm<renameModel>();

  const onSubmitRename: SubmitHandler<renameModel> = async (data) => {
    if (currentModel) {
      await renameBertModel(currentModel, data.new_name);
      resetRename();
    } else notify({ type: 'error', message: 'New name is void' });
  };

  // form to train a model
  const { trainBertModel } = useTrainBertModel(projectSlug, currentScheme);
  const {
    handleSubmit: handleSubmitNewModel,
    register: registerNewModel,
    reset: resetNewModel,
  } = useForm<newBertModel>({
    defaultValues: {
      parameters: {
        batchsize: 4,
        gradacc: 1.0,
        epochs: 3,
        lrate: 5e-5,
        wdecay: 0.01,
        best: true,
        eval: 10,
        gpu: false,
        adapt: true,
      },
    },
  });
  const onSubmitNewModel: SubmitHandler<newBertModel> = async (data) => {
    await trainBertModel(data);
    resetNewModel();
    console.log(data);
  };

  return (
    <ProjectPageLayout projectName={projectSlug} currentAction="train">
      <div className="container-fluid">
        <div className="row">
          <div className="col-2"></div>
          <div className="col-8">
            <h2 className="subsection">Models</h2>
            <span className="explanations">Train and modify models</span>
            <h4 className="subsection">Existing models</h4>
            <label htmlFor="selected-model">Existing models</label>
            <div className="d-flex align-items-center">
              <select
                id="selected-model"
                className="form-select"
                onChange={(e) => setCurrentModel(e.target.value)}
              >
                <option></option>
                {availableModels.map((e) => (
                  <option key={e}>{e}</option>
                ))}
              </select>
              <button
                className="btn btn p-0"
                onClick={() => {
                  if (currentModel) {
                    deleteBertModel(currentModel);
                  }
                }}
              >
                <MdOutlineDeleteOutline size={30} />
              </button>
            </div>

            {currentModel && (
              <div>
                <details className="custom-details">
                  <summary className="custom-summary">Rename</summary>
                  <form onSubmit={handleSubmitRename(onSubmitRename)}>
                    <input
                      id="new_name"
                      className="form-control me-2 mt-2"
                      type="text"
                      placeholder="New name of the model"
                      {...registerRename('new_name')}
                    />
                    <button className="btn btn-primary me-2 mt-2">Rename</button>
                  </form>
                </details>
                <details className="custom-details">
                  {' '}
                  <summary className="custom-summary">Description of the model</summary>
                </details>
              </div>
            )}

            <h4 className="subsection">Train a new model</h4>
            <form onSubmit={handleSubmitNewModel(onSubmitNewModel)}>
              <label htmlFor="new-model-type"></label>
              <div>
                <label>Model base</label>

                <select id="new-model-type" {...registerNewModel('base')}>
                  {(project?.bertmodels.options ? project?.bertmodels.options : []).map((e) => (
                    <option key={e}>{e}</option>
                  ))}
                </select>
              </div>
              <div>
                <label>Name to identify the model</label>
                <input type="text" {...registerNewModel('name')} placeholder="Name the model" />
              </div>
              <div>
                <label>Batch Size:</label>
                <input type="number" {...registerNewModel('parameters.batchsize')} />
              </div>
              <div>
                <label>Gradient Accumulation:</label>
                <input type="number" step="0.01" {...registerNewModel('parameters.gradacc')} />
              </div>
              <div>
                <label>Epochs:</label>
                <input type="number" {...registerNewModel('parameters.epochs')} />
              </div>
              <div>
                <label>Learning Rate:</label>
                <input type="number" step="0.00001" {...registerNewModel('parameters.lrate')} />
              </div>
              <div>
                <label>Weight Decay:</label>
                <input type="number" step="0.001" {...registerNewModel('parameters.wdecay')} />
              </div>
              <div>
                <label>Eval:</label>
                <input type="number" {...registerNewModel('parameters.eval')} />
              </div>
              <div className="form-group d-flex align-items-center">
                <label>Best:</label>
                <input type="checkbox" {...registerNewModel('parameters.best')} />
              </div>

              <div className="form-group d-flex align-items-center">
                <label>GPU:</label>
                <input type="checkbox" {...registerNewModel('parameters.gpu')} />
              </div>
              <div className="form-group d-flex align-items-center">
                <label>Adapt:</label>
                <input type="checkbox" {...registerNewModel('parameters.adapt')} />
              </div>

              <button className="btn btn-primary me-2 mt-2">Train</button>
            </form>
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
