import { FC, useState } from 'react';
import { SubmitHandler, useForm } from 'react-hook-form';
import { MdOutlineDeleteOutline } from 'react-icons/md';
import { useParams } from 'react-router-dom';

import { BertModelParametersModel } from '../types';
import { ProjectPageLayout } from './layout/ProjectPageLayout';

/**
 * Component to manage model training
 */

interface newModel {
  base: string;
  parameters: BertModelParametersModel;
}

export const ProjectTrainPage: FC = () => {
  const { projectName } = useParams();
  if (!projectName) return null;

  const [currentModel, setCurrentModel] = useState<string | null>(null);

  // form to train a model
  const { handleSubmit, register, reset } = useForm<newModel>();
  const onSubmit: SubmitHandler<newModel> = async (data) => {
    console.log(data);
  };

  return (
    <ProjectPageLayout projectName={projectName} currentAction="train">
      <div className="container-fluid">
        <div className="row">
          <div className="col-2"></div>
          <div className="col-8">
            <h2 className="subsection">Models</h2>
            <span className="explanations">Train and modify models</span>
            <h4 className="subsection">Existing models</h4>
            <label htmlFor="selected-model">Existing models</label>
            <div className="d-flex align-items-center">
              <select id="selected-model" onChange={(e) => setCurrentModel(e.target.value)}>
                <option></option>
              </select>
              <button
                className="btn btn p-0"
                onClick={() => {
                  //deleteModel(currentModel);
                  //reFetchUsers();
                  console.log('delete model');
                }}
              >
                <MdOutlineDeleteOutline size={30} />
              </button>
            </div>

            <div>
              <details>
                <summary>Rename</summary>
              </details>
              {currentModel && (
                <details>
                  {' '}
                  <summary>Description of the model</summary>
                  <button>Compute prediction</button>
                </details>
              )}
            </div>
            <h4 className="subsection">Train a new model</h4>
            <form onSubmit={handleSubmit(onSubmit)}>
              <label htmlFor="new-model-type"></label>
              <select id="new-model-type" {...register('base')}></select>
              <button className="btn btn-primary me-2 mt-2">Train</button>
            </form>
          </div>
        </div>
      </div>
    </ProjectPageLayout>
  );
};
